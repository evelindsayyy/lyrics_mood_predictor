# Eval report — transformer (distilbert-mood-v1)

- n: 7660
- accuracy: 0.5208
- macro_f1: 0.3951

| class | precision | recall |
|---|---|---|
| Angry | 0.246 | 0.412 |
| Calm | 0.448 | 0.216 |
| Hype | 0.684 | 0.676 |
| Romantic | 0.302 | 0.337 |
| Sad | 0.405 | 0.354 |

## Confusion (true → predicted)

| true \ pred | Angry | Calm | Hype | Romantic | Sad |
|---|---|---|---|---|---|
| Angry | 273 | 5 | 286 | 33 | 66 |
| Calm | 42 | 143 | 167 | 121 | 189 |
| Hype | 577 | 31 | 2827 | 428 | 320 |
| Romantic | 41 | 43 | 436 | 303 | 77 |
| Sad | 178 | 97 | 417 | 117 | 443 |
