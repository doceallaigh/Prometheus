# Corrector tap-layer sweep

Qwen2.5-0.5B-Instruct; identical seed-20260725 CfC correctors; 3,000 training steps;
GSM8K first 200 test problems; greedy and latent SC@8 at temperature 0.6.

| layer | final loss | greedy strict | greedy lenient | latent SC@8 | SC@8 internal tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1660 | 0.395 | 0.445 | 0.540 | 2397.5 |
| 1 | 0.1667 | 0.425 | 0.430 | 0.520 | 2367.5 |
| 2 | 0.1589 | 0.415 | 0.420 | 0.550 | 2365.4 |
| 3 | 0.1600 | 0.455 | 0.465 | 0.545 | 2414.0 |
| 4 | 0.1514 | 0.420 | 0.425 | 0.545 | 2365.9 |
| 5 | 0.1411 | 0.435 | 0.435 | 0.480 | 2412.9 |
| 6 | 0.1492 | 0.415 | 0.415 | 0.555 | 2384.6 |
| 7 | 0.1444 | 0.420 | 0.420 | 0.535 | 2373.6 |
| 8 | 0.1420 | 0.410 | 0.410 | 0.510 | 2383.3 |
| 9 | 0.1527 | 0.425 | 0.425 | 0.530 | 2419.9 |
| 10 | 0.1463 | 0.390 | 0.390 | 0.515 | 2361.0 |
| 11 | 0.1442 | 0.430 | 0.430 | 0.555 | 2373.3 |
| 12 | 0.1399 | 0.430 | 0.430 | 0.525 | 2419.3 |
| 13 | 0.1471 | 0.435 | 0.445 | 0.535 | 2392.8 |
| 14 | 0.1403 | 0.400 | 0.405 | 0.560 | 2413.9 |
| 15 | 0.1469 | 0.435 | 0.435 | 0.530 | 2413.5 |
| 16 | 0.1363 | 0.425 | 0.425 | 0.555 | 2342.1 |
| 17 | 0.1519 | 0.420 | 0.425 | 0.565 | 2425.2 |
| 18 | 0.1374 | 0.440 | 0.445 | 0.545 | 2430.3 |
| 19 | 0.1454 | 0.455 | 0.465 | 0.545 | 2403.6 |
| 20 | 0.1224 | 0.465 | 0.465 | 0.555 | 2380.5 |
| 21 | 0.1529 | 0.450 | 0.465 | 0.560 | 2482.6 |
| 22 | 0.1535 | 0.465 | 0.475 | 0.565 | 2444.6 |
| 23 | 0.1564 | 0.470 | 0.490 | 0.535 | 2437.7 |
| 24 | 0.1612 | 0.425 | 0.435 | 0.545 | 2491.2 |

The controlled sweep does not reproduce a unique midpoint optimum. Greedy
accuracy rises late and peaks at layer 23 (0.490 lenient), while SC@8 is broad
and irregular: layers 17 and 22 tie at 0.565, versus 0.525 at layer 12 and
0.545 at the final state. Across all taps SC@8 spans 0.480--0.565. These data
support tap robustness and a weak late-stack preference, not a sharply localized
repair layer. The original four-tap pilot remains the historical reason layer 12
was selected, but it confounded tap position with an uncontrolled initialization;
the common-seed sweep is the stronger layer-selection result.
