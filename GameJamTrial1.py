"""
Bronze Age Civ vs Rickover Civ (table-driven, turn-based)
- Assign population to jobs each turn
- Generate resources (Food/Money), production (Production Points/Isotopes), culture/research
- Buy buildings/institutions that boost yields
- Unlock tech/traits with research points / papers
- Handle random threat/pressure events
- Win by building Great Monument / founding RSI before turn 120

No external libraries required.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple
import random
import math


# -----------------------------
# Core data structures
# -----------------------------

@dataclass
class Building:
    name: str
    cost: int  # production currency (prod points / isotopes)
    desc: str
    # Flat yield bonuses added each turn after base yields
    yield_bonus: Dict[str, float] = field(default_factory=dict)
    # Multipliers applied to specific yields after flat bonuses
    yield_mult: Dict[str, float] = field(default_factory=dict)
    # Optional one-time effect on purchase
    on_buy: Optional[Callable[["Civ"], None]] = None


@dataclass
class Tech:
    name: str
    cost: int  # research currency (research points / papers)
    desc: str
    yield_bonus: Dict[str, float] = field(default_factory=dict)
    yield_mult: Dict[str, float] = field(default_factory=dict)
    unlocks_buildings: List[str] = field(default_factory=list)


@dataclass
class CivConfig:
    # Labels
    civ_name: str
    population_name: str              # People / Nerds
    surplus_resource_name: str        # Food / Money
    production_resource_name: str     # Production Points / Isotopes
    culture_resource_name: str        # Culture / Research
    research_spend_name: str          # Research points / Papers
    threat_name: str                  # Threat level / Pressure level

    # Victory
    victory_building_name: str        # Great Monument / RSI
    victory_deadline_turn: int = 120

    # Jobs: job_name -> per-worker base yields per turn
    # These yield keys must match the resource keys used internally:
    # "surplus", "prod", "culture", "threat"
    jobs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Population growth / shrink behavior
    base_surplus_consumption_per_pop: float = 1.0   # food/money upkeep per pop each turn
    pop_growth_surplus_threshold: float = 8.0       # if surplus >= threshold => +pop
    pop_loss_surplus_threshold: float = -4.0        # if surplus <= threshold => -pop

    # Threat / pressure baseline random drift
    threat_event_chance: float = 0.25

    # Starting state
    start_population: int = 6
    start_surplus: float = 8.0
    start_prod: float = 0.0
    start_culture: float = 0.0
    start_threat: float = 0.0


@dataclass
class Civ:
    cfg: CivConfig
    turn: int = 1

    pop: int = 0
    surplus: float = 0.0   # Food / Money
    prod: float = 0.0      # Production points / Isotopes
    culture: float = 0.0   # Culture / Research
    threat: float = 0.0    # Threat / Pressure

    # Allocations: job -> workers
    workers: Dict[str, int] = field(default_factory=dict)

    # Built buildings and unlocked tech
    buildings: Dict[str, int] = field(default_factory=dict)
    techs: Dict[str, bool] = field(default_factory=dict)

    # Derived modifiers
    flat_bonus: Dict[str, float] = field(default_factory=lambda: {"surplus": 0.0, "prod": 0.0, "culture": 0.0, "threat": 0.0})
    mult_bonus: Dict[str, float] = field(default_factory=lambda: {"surplus": 1.0, "prod": 1.0, "culture": 1.0, "threat": 1.0})

    def __post_init__(self):
        self.pop = self.cfg.start_population
        self.surplus = self.cfg.start_surplus
        self.prod = self.cfg.start_prod
        self.culture = self.cfg.start_culture
        self.threat = self.cfg.start_threat

        # Default allocations: all into first job
        first_job = next(iter(self.cfg.jobs.keys()))
        self.workers = {job: 0 for job in self.cfg.jobs}
        self.workers[first_job] = self.pop

    def total_workers(self) -> int:
        return sum(self.workers.values())

    def validate_workers(self) -> bool:
        return self.total_workers() == self.pop and all(v >= 0 for v in self.workers.values())

    def apply_modifiers_from_assets(self, all_buildings: Dict[str, Building], all_techs: Dict[str, Tech]) -> None:
        # Reset
        self.flat_bonus = {"surplus": 0.0, "prod": 0.0, "culture": 0.0, "threat": 0.0}
        self.mult_bonus = {"surplus": 1.0, "prod": 1.0, "culture": 1.0, "threat": 1.0}

        # Buildings
        for bname, count in self.buildings.items():
            b = all_buildings[bname]
            for k, v in b.yield_bonus.items():
                self.flat_bonus[k] += v * count
            for k, v in b.yield_mult.items():
                # stack multiplicatively: 1.10 then 1.10 => 1.21, etc.
                self.mult_bonus[k] *= (v ** count)

        # Tech
        for tname, unlocked in self.techs.items():
            if not unlocked:
                continue
            t = all_techs[tname]
            for k, v in t.yield_bonus.items():
                self.flat_bonus[k] += v
            for k, v in t.yield_mult.items():
                self.mult_bonus[k] *= v

    def compute_base_yields(self) -> Dict[str, float]:
        base = {"surplus": 0.0, "prod": 0.0, "culture": 0.0, "threat": 0.0}
        for job, n in self.workers.items():
            y = self.cfg.jobs[job]
            for k in base:
                base[k] += y.get(k, 0.0) * n
        return base

    def end_turn_update(self, all_buildings: Dict[str, Building], all_techs: Dict[str, Tech], rng: random.Random) -> None:
        # Apply modifiers
        self.apply_modifiers_from_assets(all_buildings, all_techs)

        # Base yields from workers
        base = self.compute_base_yields()

        # Upkeep
        upkeep = self.cfg.base_surplus_consumption_per_pop * self.pop

        # Total yields with modifiers
        y_surplus = (base["surplus"] + self.flat_bonus["surplus"]) * self.mult_bonus["surplus"]
        y_prod    = (base["prod"]    + self.flat_bonus["prod"])    * self.mult_bonus["prod"]
        y_culture = (base["culture"] + self.flat_bonus["culture"]) * self.mult_bonus["culture"]
        y_threat  = (base["threat"]  + self.flat_bonus["threat"])  * self.mult_bonus["threat"]

        # Threat event drift (random)
        if rng.random() < self.cfg.threat_event_chance:
            shock = rng.uniform(-2.0, 3.5)
            y_threat += shock

        # Update stocks
        self.surplus += y_surplus - upkeep
        self.prod    += y_prod
        self.culture += y_culture
        self.threat  = max(0.0, self.threat + y_threat)

        # Population changes based on surplus
        if self.surplus >= self.cfg.pop_growth_surplus_threshold:
            self.pop += 1
            self.surplus -= 2.0  # “cost” of supporting new pop this turn
        elif self.surplus <= self.cfg.pop_loss_surplus_threshold and self.pop > 1:
            self.pop -= 1
            self.surplus += 1.0  # relief

        # Rebalance allocations if pop changed (keep proportions, adjust first job for rounding)
        if self.total_workers() != self.pop:
            self._normalize_workers()

        self.turn += 1

    def _normalize_workers(self) -> None:
        # scale existing allocations to new pop, preserve ratios
        total = max(1, self.total_workers())
        desired = {job: int(round(self.workers[job] / total * self.pop)) for job in self.workers}
        # fix rounding to exact pop
        diff = self.pop - sum(desired.values())
        first_job = next(iter(self.cfg.jobs.keys()))
        desired[first_job] += diff
        # sanitize
        for job in desired:
            desired[job] = max(0, desired[job])
        # final fix if weirdness
        while sum(desired.values()) != self.pop:
            if sum(desired.values()) < self.pop:
                desired[first_job] += 1
            else:
                # remove from any job with >0
                for job in desired:
                    if desired[job] > 0:
                        desired[job] -= 1
                        break
        self.workers = desired


# -----------------------------
# Game content (table-driven)
# -----------------------------

def make_bronze_age_config() -> CivConfig:
    return CivConfig(
        civ_name="Bronze Age Civilization",
        population_name="People",
        surplus_resource_name="Food",
        production_resource_name="Production Points",
        culture_resource_name="Culture",
        research_spend_name="Research Points",
        threat_name="Threat Level",
        victory_building_name="Great Monument",
        jobs={
            # job yields per worker per turn
            "Farmers":   {"surplus": 2.2, "prod": 0.0, "culture": 0.0, "threat": 0.0},
            "Miners":    {"surplus": 0.3, "prod": 1.6, "culture": 0.0, "threat": 0.0},
            "Artisans":  {"surplus": 0.7, "prod": 0.6, "culture": 0.8, "threat": 0.0},
            "Soldiers":  {"surplus": 0.2, "prod": 0.0, "culture": 0.0, "threat": -0.8},  # lower threat
        },
    )

def make_rickover_config() -> CivConfig:
    return CivConfig(
        civ_name="Rickover Civilization",
        population_name="Nerds",
        surplus_resource_name="Money",
        production_resource_name="Isotopes",
        culture_resource_name="Research",
        research_spend_name="Papers",
        threat_name="Pressure Level",
        victory_building_name="RSI",
        jobs={
            "Fundraisers":     {"surplus": 2.1, "prod": 0.0, "culture": 0.0, "threat": 0.0},
            "Army Engineers":  {"surplus": 0.2, "prod": 1.7, "culture": 0.0, "threat": 0.0},
            "Writers":         {"surplus": 0.6, "prod": 0.4, "culture": 0.9, "threat": 0.0},
            "Recruiters":      {"surplus": 0.3, "prod": 0.0, "culture": 0.2, "threat": -0.6},  # reduce pressure
        },
    )

def make_buildings(cfg: CivConfig) -> Dict[str, Building]:
    # Names and flavor adapt to mode
    is_rickover = (cfg.civ_name == "Rickover Civilization")
    if not is_rickover:
        return {
            "Granary": Building(
                name="Granary",
                cost=18,
                desc="Reduces food volatility; better growth.",
                yield_bonus={"surplus": 1.2},
            ),
            "Quarry": Building(
                name="Quarry",
                cost=22,
                desc="Improves stone extraction.",
                yield_mult={"prod": 1.10},
            ),
            "Temple": Building(
                name="Temple",
                cost=24,
                desc="Culture engine.",
                yield_bonus={"culture": 1.2},
                yield_mult={"culture": 1.08},
            ),
            "Barracks": Building(
                name="Barracks",
                cost=20,
                desc="Stabilizes threats.",
                yield_bonus={"threat": -0.6},
            ),
            # Victory build
            "Great Monument": Building(
                name="Great Monument",
                cost=120,
                desc="Build before turn 120 to win.",
            ),
        }
    else:
        return {
            "Endowment Fund": Building(
                name="Endowment Fund",
                cost=18,
                desc="Raises steady money inflow.",
                yield_bonus={"surplus": 1.1},
                yield_mult={"surplus": 1.05},
            ),
            "Reactor Lab": Building(
                name="Reactor Lab",
                cost=22,
                desc="Boosts isotope output.",
                yield_mult={"prod": 1.12},
            ),
            "Writing Room": Building(
                name="Writing Room",
                cost=24,
                desc="Paper mill.",
                yield_bonus={"culture": 1.1},
                yield_mult={"culture": 1.10},
            ),
            "Alumni Office": Building(
                name="Alumni Office",
                cost=20,
                desc="Reduces pressure / student poaching.",
                yield_bonus={"threat": -0.6},
            ),
            # Victory build
            "RSI": Building(
                name="RSI",
                cost=120,
                desc="Found before turn 120 to win.",
            ),
        }

def make_tech_tree(cfg: CivConfig) -> Dict[str, Tech]:
    # Tech costs are paid in culture currency ("research points"/"papers")
    is_rickover = (cfg.civ_name == "Rickover Civilization")
    if not is_rickover:
        return {
            "Irrigation": Tech(
                name="Irrigation",
                cost=10,
                desc="More food per farmer.",
                yield_mult={"surplus": 1.08},
            ),
            "Bronze Tools": Tech(
                name="Bronze Tools",
                cost=12,
                desc="Miners get more production.",
                yield_mult={"prod": 1.08},
            ),
            "Written Language": Tech(
                name="Written Language",
                cost=12,
                desc="Culture compounding begins.",
                yield_mult={"culture": 1.10},
            ),
            "Standing Army": Tech(
                name="Standing Army",
                cost=14,
                desc="Threat control.",
                yield_bonus={"threat": -0.8},
            ),
        }
    else:
        return {
            "Grantwriting": Tech(
                name="Grantwriting",
                cost=10,
                desc="More money per fundraiser.",
                yield_mult={"surplus": 1.08},
            ),
            "Enrichment Methods": Tech(
                name="Enrichment Methods",
                cost=12,
                desc="More isotopes from engineers.",
                yield_mult={"prod": 1.10},
            ),
            "Publication Machine": Tech(
                name="Publication Machine",
                cost=12,
                desc="More papers from writers.",
                yield_mult={"culture": 1.12},
            ),
            "Legendary Alumni": Tech(
                name="Legendary Alumni",
                cost=14,
                desc="Pressure control + recruitment aura.",
                yield_bonus={"threat": -0.9},
            ),
        }


# -----------------------------
# UI + gameplay
# -----------------------------

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def fmt(n: float) -> str:
    # friendly formatting for resources
    if abs(n) >= 1000:
        return f"{n:,.0f}"
    return f"{n:.1f}"

def print_status(c: Civ) -> None:
    cfg = c.cfg
    print("\n" + "=" * 72)
    print(f"{cfg.civ_name}  |  Turn {c.turn}/{cfg.victory_deadline_turn}")
    print("-" * 72)
    print(f"{cfg.population_name}: {c.pop}   "
          f"{cfg.surplus_resource_name}: {fmt(c.surplus)}   "
          f"{cfg.production_resource_name}: {fmt(c.prod)}   "
          f"{cfg.culture_resource_name}: {fmt(c.culture)}   "
          f"{cfg.threat_name}: {fmt(c.threat)}")
    print("-" * 72)
    print("Assignments:")
    for job, n in c.workers.items():
        print(f"  - {job:<14} {n}")
    print("-" * 72)
    if c.buildings:
        built = ", ".join([f"{k} x{v}" if v > 1 else k for k, v in c.buildings.items()])
    else:
        built = "(none)"
    if any(c.techs.values()):
        techs = ", ".join([k for k, v in c.techs.items() if v])
    else:
        techs = "(none)"
    print(f"Buildings: {built}")
    print(f"Unlocked:  {techs}")
    print("=" * 72)

def choose_allocations(c: Civ) -> None:
    cfg = c.cfg
    print("\nReassign jobs (press Enter to keep current).")
    print(f"Total {cfg.population_name} to assign: {c.pop}")
    print("Enter counts per job. We'll auto-fix if you mis-sum.\n")

    new_workers = dict(c.workers)
    remaining = c.pop

    for i, job in enumerate(cfg.jobs.keys()):
        cur = new_workers[job]
        prompt = f"{job} (current {cur}): "
        s = input(prompt).strip()
        if s == "":
            n = cur
        else:
            try:
                n = int(s)
            except ValueError:
                n = cur

        n = max(0, n)
        new_workers[job] = n

    total = sum(new_workers.values())
    if total != c.pop:
        # normalize by scaling then adjust first job
        jobs = list(cfg.jobs.keys())
        if total <= 0:
            new_workers = {j: 0 for j in jobs}
            new_workers[jobs[0]] = c.pop
        else:
            scaled = {j: int(round(new_workers[j] / total * c.pop)) for j in jobs}
            diff = c.pop - sum(scaled.values())
            scaled[jobs[0]] += diff
            # fix negatives
            for j in jobs:
                scaled[j] = max(0, scaled[j])
            # exact fix
            while sum(scaled.values()) != c.pop:
                if sum(scaled.values()) < c.pop:
                    scaled[jobs[0]] += 1
                else:
                    for j in jobs:
                        if scaled[j] > 0:
                            scaled[j] -= 1
                            break
            new_workers = scaled
        print(f"\n(Adjusted allocations to sum to {c.pop}.)")

    c.workers = new_workers

def list_purchasables(c: Civ, buildings: Dict[str, Building], techs: Dict[str, Tech]) -> Tuple[List[str], List[str]]:
    # simple: all buildings always available, plus victory building always available
    # techs always available if not unlocked
    available_buildings = list(buildings.keys())
    available_techs = [t for t in techs.keys() if not c.techs.get(t, False)]
    return available_buildings, available_techs

def purchase_menu(c: Civ, buildings: Dict[str, Building], techs: Dict[str, Tech]) -> None:
    cfg = c.cfg
    while True:
        print("\nPurchase menu:")
        print(f"  [{cfg.production_resource_name}] = {fmt(c.prod)}  |  [{cfg.culture_resource_name}] = {fmt(c.culture)}")
        print("  1) Buy building/institution (costs production currency)")
        print("  2) Unlock tech/trait (costs culture currency)")
        print("  3) Exit menu")
        choice = input("> ").strip()

        if choice == "1":
            bnames, _ = list_purchasables(c, buildings, techs)
            print("\nBuildings / Institutions:")
            for idx, bname in enumerate(bnames, 1):
                b = buildings[bname]
                owned = c.buildings.get(bname, 0)
                own_str = f" (owned {owned})" if owned else ""
                print(f"  {idx:>2}) {b.name} — Cost {b.cost} {cfg.production_resource_name}{own_str}")
                print(f"      {b.desc}")
            s = input("\nPick # to buy (Enter to cancel): ").strip()
            if s == "":
                continue
            try:
                i = int(s)
                if not (1 <= i <= len(bnames)):
                    continue
            except ValueError:
                continue
            bname = bnames[i - 1]
            b = buildings[bname]
            if c.prod < b.cost:
                print("Not enough production currency.")
                continue
            c.prod -= b.cost
            c.buildings[bname] = c.buildings.get(bname, 0) + 1
            if b.on_buy:
                b.on_buy(c)
            print(f"Purchased: {b.name}")

        elif choice == "2":
            _, tnames = list_purchasables(c, buildings, techs)
            if not tnames:
                print("\nNo techs left to unlock.")
                continue
            print("\nTech / Skill Tree:")
            for idx, tname in enumerate(tnames, 1):
                t = techs[tname]
                print(f"  {idx:>2}) {t.name} — Cost {t.cost} {cfg.research_spend_name}")
                print(f"      {t.desc}")
            s = input("\nPick # to unlock (Enter to cancel): ").strip()
            if s == "":
                continue
            try:
                i = int(s)
                if not (1 <= i <= len(tnames)):
                    continue
            except ValueError:
                continue
            tname = tnames[i - 1]
            t = techs[tname]
            if c.culture < t.cost:
                print("Not enough culture/research currency.")
                continue
            c.culture -= t.cost
            c.techs[tname] = True
            print(f"Unlocked: {t.name}")

        elif choice == "3":
            return
        else:
            continue

def handle_event(c: Civ, rng: random.Random) -> None:
    """Flavor events influenced by threat/pressure and sometimes surplus."""
    cfg = c.cfg
    is_rickover = (cfg.civ_name == "Rickover Civilization")

    # probability increases with threat
    p = min(0.60, 0.10 + 0.02 * c.threat)
    if rng.random() > p:
        return

    if not is_rickover:
        events = [
            ("Raiders at the border",
             "You lose some food unless you stabilize with soldiers.",
             lambda: setattr(c, "surplus", c.surplus - rng.uniform(2, 6))),
            ("Festival season",
             "Culture spikes, but you burn surplus on celebration.",
             lambda: (setattr(c, "culture", c.culture + rng.uniform(2, 5)),
                      setattr(c, "surplus", c.surplus - rng.uniform(1, 3)))),
            ("Copper vein discovered",
             "A lucky production jump.",
             lambda: setattr(c, "prod", c.prod + rng.uniform(3, 8))),
        ]
    else:
        events = [
            ("Program poaches your top nerds",
             "Pressure rises; you lose some money to compete.",
             lambda: (setattr(c, "threat", c.threat + rng.uniform(2, 5)),
                      setattr(c, "surplus", c.surplus - rng.uniform(2, 6)))),
            ("Breakthrough result",
             "Papers surge.",
             lambda: setattr(c, "culture", c.culture + rng.uniform(2, 6))),
            ("Isotope batch success",
             "Production jumps.",
             lambda: setattr(c, "prod", c.prod + rng.uniform(3, 8))),
        ]

    title, desc, eff = rng.choice(events)
    eff()
    print(f"\n[Event] {title}: {desc}")

def check_victory(c: Civ) -> Optional[str]:
    cfg = c.cfg
    if c.buildings.get(cfg.victory_building_name, 0) >= 1 and c.turn <= cfg.victory_deadline_turn + 1:
        return f"WIN: You completed {cfg.victory_building_name} in time!"
    if c.turn > cfg.victory_deadline_turn + 1:
        # after resolving turn 120 -> turn becomes 121
        if c.buildings.get(cfg.victory_building_name, 0) >= 1:
            return f"WIN (late): You built {cfg.victory_building_name}, but after the deadline."
        return f"LOSS: Turn {cfg.victory_deadline_turn} passed without building {cfg.victory_building_name}."
    if c.pop <= 0:
        return "LOSS: Your civilization collapsed."
    return None

def main() -> None:
    rng = random.Random()

    print("Choose a mode:")
    print("  1) Bronze Age Civilization (Pyramid-style)")
    print("  2) Rickover Civilization (RSI-style)")
    mode = input("> ").strip()

    cfg = make_rickover_config() if mode == "2" else make_bronze_age_config()
    buildings = make_buildings(cfg)
    techs = make_tech_tree(cfg)

    civ = Civ(cfg=cfg)
    civ.techs = {k: False for k in techs.keys()}

    print(f"\nStarting: {cfg.civ_name}")
    print(f"Goal: Build {cfg.victory_building_name} by turn {cfg.victory_deadline_turn}.\n")

    while True:
        print_status(civ)

        # turn actions
        print("\nActions:")
        print("  1) Reassign jobs")
        print("  2) Purchase (buildings / tech)")
        print("  3) End turn")
        print("  4) Quit")
        a = input("> ").strip()

        if a == "1":
            choose_allocations(civ)
        elif a == "2":
            purchase_menu(civ, buildings, techs)
        elif a == "3":
            # events happen before yield update sometimes (flavor). Here: pre + post mix.
            handle_event(civ, rng)
            civ.end_turn_update(buildings, techs, rng)
            # check victory after advancing turn
            result = check_victory(civ)
            if result:
                print("\n" + result)
                print_status(civ)
                break
        elif a == "4":
            print("Goodbye.")
            break
        else:
            continue


if __name__ == "__main__":
    main()
