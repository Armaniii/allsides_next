# Junto Evidence Pipeline Frontend Display

## Current Frontend Card Display Analysis

Based on the frontend components, here's how the Junto Evidence pipeline data will be displayed:

### 1. Main Card Display (ArgumentCard.tsx)

The main card shows:
- **Stance Header**: The position/stance name (e.g., "Renewable energy should replace fossil fuels")
- **Core Argument**: A brief main argument (e.g., "Evidence-based analysis of renewable energy should replace fossil fuels")
- **"Learn More" Button**: Opens the modal with full evidence details

```
┌─────────────────────────────────────────────────────────┐
│ 🟣 Renewable energy should replace fossil fuels         │
│ ─────────────────────────────────────────────────────   │
│                                                         │
│ Evidence-based analysis of renewable energy should      │
│ replace fossil fuels                                    │
│                                                         │
│ ─────────────────────────────────────────────────────   │
│                                                         │
│        [ Learn more ⌄ ]                                 │
└─────────────────────────────────────────────────────────┘
```

### 2. Modal Display (SupportingArgumentsModal.tsx)

When users click "Learn More", the modal displays:

#### Header Section:
- **Stance Title**: Same as card header
- **Close Button (X)**

#### Core Argument Section:
- Label: "Core Argument"
- The main argument text
- Thumbs up/down rating buttons

#### Supporting Arguments Section:
- Label: "Supporting Arguments"
- List of evidence-based arguments WITH clickable domain links

### 3. Link Formatting with Junto Evidence

The `formatTextWithUrls` function (line 71-115) handles URL display:

**Current Behavior:**
- Detects URLs in text (with or without @ prefix)
- Converts them to clickable links
- Displays as: `[LinkIcon] domain.com`

**With Junto Evidence Pipeline:**
The supporting arguments will contain text like:
```
"NASA confirmed in a March 2024 press release that it is collaborating with OpenAI [nasa.gov]"
```

The function will need to be updated to handle the `[domain.com]` format. Currently it looks for full URLs, but our new format has domain-based citations.

### 4. Visual Example of Final Display

**Main Card:**
```
┌─────────────────────────────────────────────────────────┐
│ 🟣 AI Benefits for Society                              │
│ ─────────────────────────────────────────────────────   │
│                                                         │
│ Evidence-based analysis of ai benefits for society      │
│                                                         │
│ ─────────────────────────────────────────────────────   │
│                                                         │
│        [ Learn more ⌄ ]                                 │
└─────────────────────────────────────────────────────────┘
```

**Modal (when "Learn More" is clicked) - NOW WITH CLICKABLE LINKS:**
```
┌─────────────────────────────────────────────────────────┐
│ AI Benefits for Society                            [X] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Core Argument                                           │
│ ─────────────                                           │
│ Evidence-based analysis of ai benefits for society      │
│                                                         │
│                    👍     👎                            │
│                                                         │
│ • Supporting Arguments                                  │
│ ─────────────────────                                   │
│                                                         │
│ According to Stanford's 2024 AI Index Report, AI       │
│ systems have significantly improved healthcare          │
│ diagnostics accuracy by 40% [🔗 stanford.edu]          │
│                    👍     👎                            │
│                                                         │
│ The World Economic Forum reports that AI automation     │
│ has created 2.3 million new jobs in emerging tech      │
│ sectors [🔗 weforum.org]                                │
│                    👍     👎                            │
│                                                         │
│ MIT researchers found that AI-assisted education tools  │
│ improved student learning outcomes by 25% in STEM      │
│ subjects [🔗 mit.edu]                                   │
│                    👍     👎                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Note:** The [🔗 domain.com] links are now clickable and will:
- Open the full URL (e.g., https://stanford.edu/ai-report-2024) in a new tab
- Show purple color with hover effects
- Include a small link icon
- Use the URL mapping from the references array

### 5. Implementation Status ✅

**All necessary frontend updates have been implemented:**

1. **✅ Updated `formatTextWithUrls` function** to handle `[domain.com]` format:
   - ✅ Parse for `[domain.com]` patterns using regex `/\\[([^\\]]+\\.[^\\]]+)\\]/g`
   - ✅ Look up the full URL from the references array
   - ✅ Create clickable links with proper styling
   - ✅ Fallback to original URL detection for backward compatibility

2. **✅ Pass references data** to the modal:
   - ✅ Updated `ModalProps` interface to include optional `references?: any[]`
   - ✅ Updated `Stance` interface in `ArgumentsDisplay.tsx` to include `references?: any[]`
   - ✅ Pass references from `ArgumentsDisplay` to `SupportingArgumentsModal`
   - ✅ Updated `ArgumentDisplay` component to accept and use references

3. **✅ Handle new data structure**:
   - ✅ Backward compatible with existing `supporting_arguments: string[]`
   - ✅ Enhanced to support `references`, `evidence_metadata`, `detailed_evidence`
   - ✅ Frontend now properly renders `[domain.com]` citations as clickable links

### 6. Enhanced Data Structure Support

The frontend now supports both the original and enhanced data structures:

**Original Interface (still supported):**
```typescript
interface Stance {
  stance: string;
  core_argument: string;
  supporting_arguments: string[];
}
```

**Enhanced Interface (now supported):**
```typescript
interface JuntoEvidenceStance {
  stance: string;
  core_argument: string;
  supporting_arguments: string[];  // Contains "[domain.com]" formatted text
  references?: Reference[];        // Contains URL mappings for clickable links
  evidence_metadata?: {...};       // Additional metadata (not yet displayed)
  detailed_evidence?: {...};       // Supporting/refuting breakdown (not yet displayed)
}
```

**Currently Implemented:**
- ✅ Parse and render clickable `[domain.com]` links using references array
- ✅ Backward compatibility with existing format
- ✅ Proper link styling with purple theme and hover effects

**Future Enhancements (not yet implemented):**
- Display evidence metadata (counts, source types)
- Show supporting vs. refuting evidence separately
- Enhanced visual indicators for evidence strength