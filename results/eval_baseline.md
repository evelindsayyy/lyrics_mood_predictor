# Eval report — baseline (baseline-lr-v1)

- n: 7660
- accuracy: 0.4316
- macro_f1: 0.3706

| class | precision | recall |
|---|---|---|
| Angry | 0.228 | 0.448 |
| Calm | 0.274 | 0.391 |
| Hype | 0.731 | 0.462 |
| Romantic | 0.260 | 0.404 |
| Sad | 0.331 | 0.362 |

## Confusion (true → predicted)

| true \ pred | Angry | Calm | Hype | Romantic | Sad |
|---|---|---|---|---|---|
| Angry | 297 | 35 | 166 | 54 | 111 |
| Calm | 44 | 259 | 93 | 105 | 161 |
| Hype | 712 | 262 | 1933 | 731 | 545 |
| Romantic | 58 | 135 | 243 | 364 | 100 |
| Sad | 190 | 255 | 210 | 144 | 453 |
