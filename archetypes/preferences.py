"""Archetype preference specification."""

CONTRIB = {
    "A1": dict(lo=(50, 40), hi=(100, 80)),   # EU27
    "A2": dict(lo=(25, 10), hi=(50, 30)),    # other high-income
    "A4": dict(lo=(20, 10), hi=(48, 30)),    # China
}

ASSUMED_CONTRIB = {
    "A3": dict(lo=(8, 8), hi=(12, 12)),      # US
    "A5": dict(lo=(6, 6), hi=(10, 10)),      # fossil exporters
}

BENEF = {
    "A6": dict(lo=0.48, hi=0.25),            # India & developing
    "A7": dict(lo=0.40, hi=0.20),            # less wealthy
}

C_LO, C_HI = 0.50, 0.85
