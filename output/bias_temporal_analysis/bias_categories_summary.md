# Bias Categories Summary

## Bias Categories Used in Analysis

The following **5 categories** are grouped together as "bias categories" and used to calculate the bias score:

1. **`Comment_ask a rhetorical question`** - Rhetorical questions used to express bias indirectly
2. **`Perception_not in my backyard`** - NIMBY attitudes
3. **`Perception_harmful generalization`** - Harmful generalizations about homelessness
4. **`Perception_deserving/undeserving`** - Deserving/undeserving framing
5. **`Racist_Flag`** - Racist content

### Bias Score Calculation

The bias score is calculated as the **sum** of these five categories for each text entry:
- If a text has 0 bias categories → bias_score = 0
- If a text has 1 bias category → bias_score = 1
- If a text has 2 bias categories → bias_score = 2
- If a text has 3 bias categories → bias_score = 3
- If a text has 4 bias categories → bias_score = 4
- If a text has 5 bias categories → bias_score = 5

## Non-Bias Categories (Indicators)

The following categories are **not** included in bias calculations but are tracked separately:

### Comment Types (6 categories):
- `Comment_ask a genuine question`
- `Comment_ask a rhetorical question` ⚠️ **NOW INCLUDED IN BIAS SCORE**
- `Comment_provide a fact or claim` ⚠️ Indicator category
- `Comment_provide an observation` ⚠️ Indicator category
- `Comment_express their opinion` ⚠️ Indicator category
- `Comment_express others opinions` ⚠️ Indicator category

### Critique Categories (3 categories):
- `Critique_money aid allocation`
- `Critique_government critique`
- `Critique_societal critique`

### Response Categories (1 category):
- `Response_solutions/interventions`

### Other Perception Types (2 categories):
- `Perception_personal interaction`
- `Perception_media portrayal`

## City Classification

### Large Cities (Similar to San Francisco, CA)
- **San Francisco** (San Francisco County, CA)
- **Portland** (Multnomah County, OR)
- **Buffalo** (Erie County, NY)
- **Baltimore** (Baltimore County, MD)
- **El Paso** (El Paso County, TX)

### Small Cities (Similar to South Bend, IN)
- **South Bend** (St. Joseph County, IN)
- **Rockford** (Winnebago County, IL)
- **Kalamazoo** (Kalamazoo County, MI)
- **Scranton** (Lackawanna County, PA)
- **Fayetteville** (Washington County, AR)

## Notes

- The bias score represents the **presence** of bias categories, not the intensity
- Multiple bias categories can co-occur in a single text
- The analysis compares average bias scores across sources and city sizes
- Statistical significance is tested using Bonferroni correction for multiple comparisons
