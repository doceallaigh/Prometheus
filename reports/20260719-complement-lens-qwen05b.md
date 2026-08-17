# Complement lens: decoding the Jacobian split with the trunk's upper half

Model: `Qwen/Qwen2.5-0.5B-Instruct`, basis: `outputs/retrofit-qwen05b/jspace-basis-rank64.pt`, tap layer: 12, dump: `reports/20260716-retrofit-qwen05b-corrector7k.completions.jsonl`, problems: 200
Plumbing sanity (full-stream decode vs teacher-forced tokens): 0.9554 argmax agreement

| stream | agree w/ full | digit rate | digit @ digit pos | contending digit | word rate | contextual number | runner-up align |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 1.0000 | 0.1565 | 1.0000 | 0.0000 | 0.4710 | 0.7142 | 0.0000 |
| dominant | 0.1042 | 0.1004 | 0.5440 | 0.3400 | 0.3014 | 0.7393 | 0.0548 |
| complement | 0.6530 | 0.1494 | 0.9081 | 0.2575 | 0.4652 | 0.7696 | 0.3453 |
| complement-parallel | 0.9669 | 0.1561 | 0.9948 | 0.0127 | 0.4713 | 0.7149 | 0.8550 |
| complement-perp | 0.0095 | 0.0691 | 0.0136 | 0.0117 | 0.1574 | 0.5599 | 0.0083 |
| random | 0.0110 | 0.0399 | 0.0396 | 0.0333 | 0.3838 | 0.5402 | 0.0087 |
| shuffled-complement | 0.0269 | 0.1416 | 0.1631 | 0.1228 | 0.4239 | 0.7919 | 0.0161 |

Digit positions (mainline): 8879 of 56735

## Decoded-token categories at mainline-digit positions

| stream | digit | operator | word | space | other |
| --- | --- | --- | --- | --- | --- |
| full | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dominant | 0.5440 | 0.3027 | 0.1301 | 0.0214 | 0.0018 |
| complement | 0.9081 | 0.0574 | 0.0089 | 0.0256 | 0.0000 |
| complement-parallel | 0.9948 | 0.0016 | 0.0003 | 0.0028 | 0.0005 |
| complement-perp | 0.0136 | 0.2968 | 0.3592 | 0.1722 | 0.1582 |
| random | 0.0396 | 0.3882 | 0.4503 | 0.0765 | 0.0454 |
| shuffled-complement | 0.1631 | 0.2572 | 0.4017 | 0.1656 | 0.0124 |

## Contending-number examples (complement decode at mainline-digit positions)

- p0 `....
   - She eats 3` mainline=`3` complement=`1`
- p0 `...   - She bakes muff4` mainline=`4` complement=`3`
- p0 `... laid per day is:
16` mainline=`6` complement=`4`
- p0 `... is:
16 - 3` mainline=`3` complement=`1`
- p0 `... + 4 = 23` mainline=`3` complement=`0`
- p0 `... number of eggs laid is 2` mainline=`2` complement=`1`
- p0 `... of eggs laid is 23` mainline=`3` complement=`0`
- p0 `...: \(23 - 3` mainline=`3` complement=`1`
- p0 `...3 - 3 - 4` mainline=`4` complement=`1`
- p0 `...3 - 4 = 1` mainline=`1` complement=`2`
- p0 `... - 4 = 16` mainline=`6` complement=`0`
- p0 `...\) eggs sold.

 day.

3` mainline=`3` complement=`2`
- p0 `... total earnings per day are \(1` mainline=`1` complement=`2`
- p0 `... earnings per day are \(16` mainline=`6` complement=`2`
- p0 `... \times 2 = 3` mainline=`3` complement=`1`
- p1 `... required is:
 \frac{2` mainline=`2` complement=`1`
- p1 `...{ (white)} = 3` mainline=`3` complement=`2`
- p1 `... number of bolts required is **3` mainline=`3` complement=`1`
- p1 `... numeric answer is:

#### 3` mainline=`3` complement=`2`
- p2 `... - Cost of repairs: $5` mainline=`5` complement=`8`
- p2 `...,000 + 5` mainline=`5` complement=`8`
- p2 `... the new value is 15` mainline=`5` complement=`0`
- p2 `... = Original value + (15` mainline=`5` complement=`0`
- p2 `...{New value} = 8` mainline=`8` complement=`1`
- p2 `...1.5 \times 8` mainline=`8` complement=`1`
- p2 `...,000 = 2` mainline=`2` complement=`8`
- p2 `...text{Profit} = 2` mainline=`2` complement=`1`
- p2 `...,000 = 7` mainline=`7` complement=`1`
- p3 `...3 = 9\).

3` mainline=`3` complement=`2`
- p3 `... is \(9 \times 6` mainline=`6` complement=`3`
- p3 `... runs in one week is **5` mainline=`5` complement=`1`
- p3 `...40 <answer>58` mainline=`8` complement=`3`
- p4 `... chickens and each chicken needs 3` mainline=`3` complement=`2`
- p4 `... needed per day} = 2` mainline=`2` complement=`3`
- p4 `... cups/chicken} = 6` mainline=`6` complement=`2`
- p4 `...   - Each meal requires 1` mainline=`1` complement=`3`
- p4 `... - Each meal requires 15` mainline=`5` complement=`0`
- p4 `... for final meal} = 1` mainline=`1` complement=`6`
- p5 `... one second glass = $ 5` mainline=`5` complement=`6`
- p5 `... $ 5 \times 0` mainline=`0` complement=`1`
