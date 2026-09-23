# Arithmetic check: profitable region report

The calculation implements the interval rule and quadratic scoring directly from
`data/actors_baseline.csv`; it does not call `elicit.rules`,
`elicit.formats`, or `elicit.measures`.

## Case

- Scenario peak fraction: `0.50`
- Rule: `region: intersection-midpoint`
- Deviator: `HYDROCARBON RENTIERS`
- True quadratic: `1 - ((p - 1.737287) / (3.474575 - 1.737287))^2`
- True acceptable interval: `[0.000000, 3.474575]`

## Truthful profile

- Selected price: `3.474575`
- Deviator participates: `True`
- True utility (zero if out): `0.000000`

## Constructive deviation

- Reported interval: `[0.779622, 1.801106]`
- Greatest-weight common intersection selected by the rule:
  `[0.779622, 1.801106]`
- Realised price after weighted-midpoint projection: `1.801106`
- Report-implied members: CHINA, UNITED STATES, EUROPEAN UNION, INDIA, RUSSIA, INDONESIA, ADV. CARBON-PRICED CONDITIONAL JOINERS, LOW-CARBON FRONTIER, HYDROCARBON RENTIERS
- Deviator participates: `True`
- True utility at realised price: `0.998651`
- Profitable gain: `0.998651 - 0.000000 = 0.998651`

The recomputed price, membership, utility, and regret agree with the saved
search output to better than `1e-8`. This profitable deviation is sufficient to
show that the region rule is not strategy-proof; it does not require the search
to have found the global optimum.
