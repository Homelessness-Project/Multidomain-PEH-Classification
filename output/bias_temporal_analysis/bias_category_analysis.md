# Bias Category Analysis: Should Additional Categories Be Included?

## Current Bias Categories (4 categories)

1. **`Perception_not in my backyard`** - NIMBY attitudes
2. **`Perception_harmful generalization`** - Harmful generalizations about homelessness
3. **`Perception_deserving/undeserving`** - Deserving/undeserving framing
4. **`Racist_Flag`** - Racist content

## Proposed Additional Categories

### 1. `Comment_ask a rhetorical question`
**Definition**: The speaker asks a question not intended to be answered, often to make a point

**Analysis**:
- ✅ **Potentially biased**: Rhetorical questions are often used to dismiss, undermine, or mock arguments
- ✅ **Examples**: "Why should we help them?" "Don't they have enough already?"
- ⚠️ **Consideration**: Not all rhetorical questions are biased - some may be used for legitimate critique
- **Recommendation**: **INCLUDE** - Rhetorical questions are frequently used as a rhetorical device to express bias without making direct statements

### 2. `Critique_government critique`
**Definition**: Criticism of government policies, laws, or political approaches to homelessness

**Analysis**:
- ⚠️ **Not inherently biased**: Government critique can be legitimate policy discussion
- ⚠️ **Context-dependent**: Could be biased if used to dismiss all government efforts or promote harmful narratives
- ❌ **Too broad**: This category includes both constructive criticism and biased attacks
- **Recommendation**: **EXCLUDE** - Government critique is too broad and includes legitimate policy discussion. However, if combined with other bias indicators, it could signal biased framing.

### 3. `Critique_societal critique`
**Definition**: Criticism of social norms, systems, or societal attitudes toward homelessness

**Analysis**:
- ⚠️ **Not inherently biased**: Societal critique can be legitimate social commentary
- ⚠️ **Context-dependent**: Could be biased if used to blame society without acknowledging systemic issues
- ❌ **Too broad**: This category includes both constructive social commentary and biased blaming
- **Recommendation**: **EXCLUDE** - Similar to government critique, this is too broad. However, societal critique that blames "society" for homelessness without acknowledging systemic factors could be biased.

## Other Categories to Consider

### Potential Bias Indicators:

1. **`Perception_media portrayal`** - Could indicate bias if media is blamed for "misrepresenting" homelessness
   - **Recommendation**: **EXCLUDE** - Media portrayal discussion is neutral; the bias comes from HOW it's discussed

2. **`Comment_express others opinions`** - Could indicate bias if used to amplify negative stereotypes
   - **Recommendation**: **EXCLUDE** - Expressing others' opinions is neutral; the bias comes from WHICH opinions are expressed

3. **`Critique_money aid allocation`** - Could indicate bias if used to argue against funding
   - **Recommendation**: **EXCLUDE** - Discussion of funding is legitimate policy discourse

## Recommended Bias Category Set

### Option 1: Conservative (Current + Rhetorical Questions)
**5 categories**:
1. `Perception_not in my backyard`
2. `Perception_harmful generalization`
3. `Perception_deserving/undeserving`
4. `Racist_Flag`
5. `Comment_ask a rhetorical question` ⭐ NEW

**Rationale**: Focuses on clearly biased content. Rhetorical questions are frequently used as a rhetorical device to express bias.

### Option 2: Moderate (Current + Rhetorical Questions + Contextual Critique)
**6 categories**:
1. `Perception_not in my backyard`
2. `Perception_harmful generalization`
3. `Perception_deserving/undeserving`
4. `Racist_Flag`
5. `Comment_ask a rhetorical question` ⭐ NEW
6. `Critique_government critique` ⭐ NEW (if combined with other bias indicators)

**Rationale**: Includes rhetorical questions and government critique when it appears alongside other bias indicators. However, this requires more complex scoring logic.

### Option 3: Broad (All Proposed)
**7 categories**:
1. `Perception_not in my backyard`
2. `Perception_harmful generalization`
3. `Perception_deserving/undeserving`
4. `Racist_Flag`
5. `Comment_ask a rhetorical question` ⭐ NEW
6. `Critique_government critique` ⭐ NEW
7. `Critique_societal critique` ⭐ NEW

**Rationale**: Most inclusive, but may over-count bias by including legitimate critique.

## Recommendation

**Recommend Option 1**: Add `Comment_ask a rhetorical question` to the bias categories.

**Reasoning**:
- Rhetorical questions are frequently used as a rhetorical device to express bias
- They allow speakers to make biased points without direct statements
- Examples like "Why should we help them?" clearly indicate bias
- This maintains focus on clearly biased content while expanding coverage

**Do NOT add government critique or societal critique** because:
- They include legitimate policy and social commentary
- They are too broad and context-dependent
- Adding them would dilute the bias score with non-biased content
- They can be analyzed separately as "critique patterns" rather than bias indicators

## Implementation Note

If adding rhetorical questions, consider:
- The bias score will now range from 0-5 instead of 0-4
- Update all documentation and visualizations
- Update the subtitle in charts to include "Rhetorical question"
