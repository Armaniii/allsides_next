# Junto Evidence Pipeline Data Models

This document describes the new Pydantic data models defined for the Junto Evidence pipeline in `main_v3.py` around line 1865.

## Model Hierarchy

The new data models are designed to handle the enhanced evidence structure with clickable domain-based links and detailed evidence metadata.

### Core Evidence Models

#### `EvidenceItem`
Individual evidence item from Junto Evidence Finder API
```python
class EvidenceItem(BaseModel):
    quote: str                    # The evidence quote text
    citation_id: int             # Reference ID for citation linking  
    reasoning: str               # Explanation of how evidence relates to claim
    stance: str                  # "supports" or "refutes"
    url: str                     # Source URL for the evidence
    domain: str                  # Extracted domain name for display
    formatted: str               # Formatted text like "Quote [nasa.gov]"
```

#### `Reference`
Reference/citation with clickable domain-based format
```python
class Reference(BaseModel):
    id: int                      # Reference ID number
    title: str                   # Display title (domain name like "nasa.gov")
    url: str                     # Full URL for the reference
    source_type: str             # "primary" or "secondary"
    stance: str                  # "supports" or "refutes"
    domain: str                  # Extracted domain name
```

### Metadata Models

#### `EvidenceMetadata`
Metadata about evidence collection for a stance
```python
class EvidenceMetadata(BaseModel):
    supporting_evidence_count: int    # Number of supporting evidence items
    refuting_evidence_count: int      # Number of refuting evidence items
    total_evidence_count: int         # Total evidence items processed
    primary_sources: int              # Number of primary source citations
    secondary_sources: int            # Number of secondary source citations
```

#### `DetailedEvidence`
Detailed evidence breakdown for future UI enhancements
```python
class DetailedEvidence(BaseModel):
    supporting: List[EvidenceItem]    # Supporting evidence items
    refuting: List[EvidenceItem]      # Refuting evidence items
```

#### `PipelineMetadata`
Metadata about the Junto Evidence pipeline execution
```python
class PipelineMetadata(BaseModel):
    positions_generated: int          # Positions from Junto Position API
    evidence_searches: int            # Number of evidence searches performed
    total_evidence_items: int         # Total evidence items collected
    total_evidence_cost: float        # Cost estimate from Junto Evidence API
    evidence_structure_version: str   # Version marker for compatibility
    link_format: str                  # "domain_based_clickable"
    citation_style: str               # "inline_domain_links"
```

### Composite Models

#### `JuntoEvidenceStance`
Enhanced stance model for Junto Evidence pipeline
```python
class JuntoEvidenceStance(BaseModel):
    stance: str                       # The stance or perspective on the topic
    core_argument: str                # Main argument supporting this stance
    supporting_arguments: List[str]   # Arguments with clickable citations like [nasa.gov]
    references: List[Reference]       # All references for clickable link mapping
    evidence_metadata: EvidenceMetadata      # Metadata about evidence collection
    detailed_evidence: DetailedEvidence      # Detailed evidence breakdown
```

#### `JuntoEvidenceResponse` (Top-Level)
Complete response model for the Junto Evidence pipeline
```python
class JuntoEvidenceResponse(BaseModel):
    arguments: List[JuntoEvidenceStance]    # List of stances with evidence
    model: str                              # "junto-evidence-pipeline"
    search_model: str                       # "junto-position-finder + junto-evidence-finder"
    references: List[Reference]             # All references for the response
    pipeline_metadata: PipelineMetadata    # Pipeline execution metadata
```

## Usage in the Pipeline

The models are used in the `complete_with_junto_evidence_pipeline()` function:

1. **Evidence Processing**: Raw evidence from Junto API is validated and structured into `EvidenceItem` objects
2. **Reference Creation**: URLs are processed into `Reference` objects with domain-based titles
3. **Metadata Generation**: Evidence statistics are captured in `EvidenceMetadata`
4. **Response Assembly**: All components are combined into a `JuntoEvidenceResponse`

## Key Features

### Clickable Domain Links
- Supporting arguments contain text like: `"NASA confirmed collaboration [nasa.gov]"`
- The `[domain.com]` parts should be rendered as clickable links by the frontend
- Use the `references` array to map domain names to full URLs

### Evidence Separation
- Supporting and refuting evidence are separated for potential UI enhancements
- Evidence metadata provides counts and source type breakdowns
- Detailed evidence is stored for future features

### Validation
- All models include Pydantic validation with field descriptions
- Type checking ensures data integrity
- Graceful fallback to dictionary format if validation fails

## Frontend Implementation

To implement clickable links in the frontend:

1. Parse `supporting_arguments` for `[domain.com]` patterns
2. Look up the corresponding `Reference` object by domain
3. Replace `[domain.com]` with clickable links to the full URL
4. Style citation links appropriately

Example:
- Text: `"NASA confirmed collaboration [nasa.gov]"`
- Replace `[nasa.gov]` with link to `https://nasa.gov/news/openai-2024`

## Comparison to Standard Pipeline

| Feature | Standard Pipeline | Junto Evidence Pipeline |
|---------|------------------|------------------------|
| Data Model | `ArgumentResponse` | `JuntoEvidenceResponse` |
| Citations | Numbered `[1]` | Domain-based `[nasa.gov]` |
| Evidence Detail | Basic arguments | Supporting/refuting separation |
| Source Types | N/A | Primary/secondary distinction |
| Link Format | Standard references | Clickable domain links |

This enhanced structure provides a foundation for rich evidence display while maintaining compatibility with existing systems.