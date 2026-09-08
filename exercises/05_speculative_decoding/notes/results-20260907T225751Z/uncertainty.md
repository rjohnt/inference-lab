# Throughput variability

95% paired bootstrap intervals for the ratio of medians, resampling repetition blocks 10,000 times. These describe timing variation in this six-prompt suite on this host; they do not cover workload diversity or systematic host interference.

| Prompt | Method | N | Median TPS | Min–max TPS | Speedup | 95% interval |
|---|---|---:|---:|---:|---:|---:|
| chat_short | none | 10 | 79.53 | 78.98–79.91 | 1.000× | 1.000–1.000× |
| chat_short | draft-eagle3 | 10 | 110.58 | 109.73–112.03 | 1.390× | 1.389–1.398× |
| chat_short | draft-dflash | 10 | 85.51 | 84.26–85.88 | 1.075× | 1.070–1.078× |
| code_512 | none | 10 | 77.36 | 76.97–77.78 | 1.000× | 1.000–1.000× |
| code_512 | draft-eagle3 | 10 | 121.41 | 120.63–121.89 | 1.569× | 1.566–1.572× |
| code_512 | draft-dflash | 10 | 118.97 | 118.47–119.29 | 1.538× | 1.534–1.541× |
| code_8192 | none | 10 | 67.26 | 66.66–67.56 | 1.000× | 1.000–1.000× |
| code_8192 | draft-eagle3 | 10 | 91.99 | 91.23–92.56 | 1.368× | 1.364–1.376× |
| code_8192 | draft-dflash | 10 | 81.75 | 81.18–82.11 | 1.216× | 1.211–1.221× |
| document_2048 | none | 10 | 75.60 | 75.45–76.02 | 1.000× | 1.000–1.000× |
| document_2048 | draft-eagle3 | 10 | 97.80 | 97.49–98.44 | 1.294× | 1.291–1.296× |
| document_2048 | draft-dflash | 10 | 107.00 | 106.21–107.41 | 1.415× | 1.407–1.418× |
| document_512 | none | 10 | 78.36 | 78.18–78.73 | 1.000× | 1.000–1.000× |
| document_512 | draft-eagle3 | 10 | 93.94 | 93.31–94.64 | 1.199× | 1.195–1.201× |
| document_512 | draft-dflash | 10 | 91.41 | 77.77–91.68 | 1.167× | 1.159–1.169× |
| document_8192 | none | 10 | 67.38 | 67.02–67.78 | 1.000× | 1.000–1.000× |
| document_8192 | draft-eagle3 | 10 | 81.44 | 80.28–81.73 | 1.209× | 1.204–1.211× |
| document_8192 | draft-dflash | 10 | 61.45 | 57.54–61.71 | 0.912× | 0.907–0.914× |
