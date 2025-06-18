## 1) Ensure the initial query is formatted as a question
    - If the query is a single topic, use local lm client to transform into a general question. i.e., if the query = "abortion" then the local lm client should transform it an appropriate question that is general, query = "is abortion good or bad?" 
    - If its already framed in a good specific question then return
    - This should be very quick call as soon as the user hits enter and should happen synchronously. 
    
## 2) Use local llm to generate a list of 4-5 amazing follow-up questions for the user to ask. 

    - Now that they have received a bunch of information about the question they were interested in, reccommend several diverse follow-up questions that will broaden their perspectives.
    -- Input should be the positions generated. 
    - These questions can be more specific questions surrounding their question and also periphery questions that are important or relevant to know in the broad discourse surrounding it
    - These suggestions should pop up in an aesthetic and slightly transparent bars (search style) above the search bar and can when clicked trigger a new search. 
    - Perform this call right after the positions are generated, it should be processed async in the background while the rest of program functions

## 3) Generate overarching summary that is 2-3 sentences and is shown as the 'core-argument' right below the position generated. 

    - The "core-argument" should be a local llm summarization of all the raw supporting and refuting arguments as well as the "reasoning" which is returned. 
    - This should happen asynchronously as each position finishes generating all the pieces of evidence. The information should be updated as efficiently as possible and integrate properly into the expected data model and visualization on the frontend. 

## 4) Provide an insightful URL analysis:
- class SourceCategorizer:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.categories = {
            'academic': ['peer-reviewed', 'university', 'research institute'],
            'news_media': ['mainstream', 'alternative', 'trade publication'],
            'advocacy': ['environmental', 'industry', 'policy think tank'],
            'government': ['agency', 'regulatory', 'international org'],
            'commercial': ['company blog', 'industry report', 'marketing'],
            'social_media': ['platform', 'influencer', 'community'],
            'independent': ['personal blog', 'wiki', 'forum']
        }
    
    def analyze_source(self, url, domain):
        prompt = f"""
        Analyze this source: {domain}
        URL: {url}
        
        Return JSON with:
        1. primary_category: (academic/news_media/advocacy/etc)
        2. credibility_score: (0-100)
        3. bias_indicators: []
        4. expertise_level: (general/specialized/expert)
        5. funding_transparency: (clear/unclear/hidden)
        """
        return self.llm.generate(prompt)

    - # Extract domain characteristics
        domain_analysis = llm.analyze(f"""
        Domain: {ref['domain']}
        
        Determine:
        1. Organization type (news/advocacy/academic/commercial)
        2. Geographic origin and scope
        3. Known affiliations or funding sources
        4. Historical reliability rating
        5. Target audience
        """)
        
        # Quantify content characteristics
        content_analysis = llm.analyze(f"""
        Based on URL: {ref['url']}
        
        Assess:
        1. Content depth (surface/moderate/deep)
        2. Citation density (none/low/medium/high)
        3. Author expertise indicators
        4. Commercial influence markers
        5. Emotional language score (0-100)
        """)
        
        ref['categorization'] = {
            'domain': domain_analysis,
            'content': content_analysis,
            'composite_trust_score': calculate_trust_score(domain_analysis, content_analysis)
        }
        
        enhanced_refs.append(ref)
    
    return enhanced_refs

    - develop a visualization for each cards sources and have it integrate extremely nicely into the card:
    Source Distribution Analysis:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    News Media:        ████████ 40% (2 sources)
    Advocacy:          ████████ 40% (2 sources)  
    Academic:          ░░░░░░░░ 0%  ⚠️
    Government:        ░░░░░░░░ 0%  ⚠️
    Industry Analysis: ████     20% (1 source)

    Trust Score Distribution:
    High (80-100):  ██ 1 source
    Medium (60-79): ████ 2 sources  
    Low (0-59):     ████ 2 sources

    Geographic Bias: 100% Western sources ⚠️
    Temporal Bias: All sources from 2024


    - This local lm anaylsis should happen asynchronously after each card finishes and then efficiently integrate the response (properly formatted) into the data model such that it can be displayed effectively on the frontend (UI/UX integration)

    UI/UX Proposal to follow for URL analysis:
        ## 1. **Minimalist Source Quality Badge**

        ```css
        /* Top-right corner indicator */
        .source-quality-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        width: 40px;
        height: 40px;
        }
        ```

        ```svg
        <!-- Circular badge showing source diversity -->
        <svg viewBox="0 0 40 40">
        <!-- Outer ring: Source diversity -->
        <circle cx="20" cy="20" r="18" 
            stroke-dasharray="28.27 84.82"  <!-- 33% diverse -->
            stroke="#10b981" stroke-width="3" fill="none"
            transform="rotate(-90 20 20)"/>
        
        <!-- Inner circle: Average credibility -->
        <circle cx="20" cy="20" r="12" fill="#f59e0b" opacity="0.8"/>
        
        <!-- Center text: Score -->
        <text x="20" y="24" text-anchor="middle" 
            font-size="11" font-weight="bold" fill="white">72</text>
        </svg>
        ```

        ## 2. **Expandable Source Bar Design**

        ```javascript
        // Collapsed state (default)
        <div className="source-bar collapsed">
        <div className="source-summary">
            <span className="source-count">5 sources</span>
            <div className="mini-distribution">
            <span className="dot news" title="2 news"></span>
            <span className="dot advocacy" title="2 advocacy"></span>
            <span className="dot blog" title="1 blog"></span>
            </div>
            <span className="trust-indicator">⚡ 72% trust</span>
        </div>
        </div>

        // Expanded state (on hover/click)
        <div className="source-bar expanded">
        <div className="source-grid">
            <div className="source-tile high-trust">Reuters</div>
            <div className="source-tile medium-trust">Forbes</div>
            <div className="source-tile low-trust warning">Blog</div>
        </div>
        </div>
        ```

        ## 3. **Clean Card Layout with Integrated Metrics**

        ```jsx
        function ArgumentCard({ data }) {
        return (
            <div className="argument-card">
            {/* Header Section */}
            <header>
                <h3>{data.core_argument}</h3>
                <SourceQualityBadge score={72} diversity={33} />
            </header>

            {/* Main Content */}
            <div className="supporting-points">
                {data.supporting_arguments.map(arg => (
                <div className="point">
                    <p>{arg.text}</p>
                    <SourceIndicator type={arg.source_type} trust={arg.trust} />
                </div>
                ))}
            </div>

            {/* Footer Metrics Bar */}
            <footer className="metrics-bar">
                <div className="metric">
                <Icon name="shield" />
                <span>72% verified</span>
                </div>
                <div className="metric warning">
                <Icon name="alert" />
                <span>Missing: Academic</span>
                </div>
                <ExpandButton />
            </footer>
            </div>
        );
        }
        ```

        ## 4. **Visual Hierarchy Approach**

        ```css
        /* Primary: The argument itself */
        .core-argument {
        font-size: 18px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 16px;
        }

        /* Secondary: Supporting points */
        .supporting-point {
        font-size: 14px;
        color: #374151;
        padding-left: 20px;
        border-left: 2px solid #e5e7eb;
        }

        /* Tertiary: Source indicators (subtle) */
        .source-indicator {
        display: inline-block;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        margin-left: 6px;
        /* Color codes: green=academic, blue=news, orange=advocacy */
        }
        ```

        ## 5. **Progressive Disclosure Pattern**

        ```jsx
        // Level 1: Just a colored dot
        <span className="source-quality high" />

        // Level 2: Hover tooltip
        <Tooltip content="Reuters • News • High credibility">
        <span className="source-quality high" />
        </Tooltip>

        // Level 3: Click for full analysis
        <Modal>
        <SourceAnalysis>
            <RadarChart metrics={['Credibility', 'Diversity', 'Recency', 'Balance']} />
            <MissingPerspectives list={['Academic', 'Government', 'Opposition']} />
        </SourceAnalysis>
        </Modal>
        ```

        ## 6. **Micro-Visualizations**

        ```jsx
        // Tiny sparkline showing source distribution
        function SourceSparkline({ sources }) {
        return (
            <svg width="60" height="16" className="source-sparkline">
            {sources.map((source, i) => (
                <rect
                key={i}
                x={i * 12}
                y={16 - source.trust * 0.16}
                width="10"
                height={source.trust * 0.16}
                fill={getColorByType(source.type)}
                opacity="0.8"
                />
            ))}
            </svg>
        );
        }
        ```

        ## 7. **Smart Warning System**

        ```jsx
        // Only show warnings that matter
        function BiasWarning({ sources }) {
        const criticalIssues = detectCriticalBias(sources);
        
        if (!criticalIssues.length) return null;
        
        return (
            <div className="bias-warning subtle">
            <Icon name="info" size="12" />
            <span>{criticalIssues[0]}</span> {/* Show only most critical */}
            </div>
        );
        }
        ```

        ## 8. **Complete Card Design Example**

        ```css
        .argument-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        position: relative;
        }

        /* Subtle gradient background for trust score */
        .argument-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(
            to right,
            #10b981 0%,    /* High trust sources */
            #f59e0b 40%,   /* Medium trust */
            #ef4444 100%   /* Low trust */
        );
        border-radius: 8px 8px 0 0;
        opacity: 0.8;
        }

        /* Clean source pills */
        .source-pills {
        display: flex;
        gap: 4px;
        margin-top: 8px;
        }

        .source-pill {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        background: #f3f4f6;
        color: #6b7280;
        }

        .source-pill.verified {
        background: #d1fae5;
        color: #065f46;
        }
        ```

        ## 9. **Mobile-First Considerations**

        ```jsx
        // Adaptive information density
        function AdaptiveSourceInfo({ sources, viewport }) {
        if (viewport === 'mobile') {
            // Ultra-minimal: just a trust score
            return <TrustBadge score={calculateAverage(sources)} />;
        }
        
        if (viewport === 'tablet') {
            // Moderate: score + warning
            return (
            <>
                <TrustBadge score={calculateAverage(sources)} />
                {hasCriticalBias(sources) && <WarningDot />}
            </>
            );
        }
        
        // Desktop: Full mini-visualization
        return <SourceSparkline sources={sources} />;
        }
        ```

        ## 10. **Final Integration Pattern**

        ```jsx
        <ArgumentCard>
        {/* Glanceable: 2-second scan */}
        <div className="at-a-glance">
            <h3>Renewable energy increasingly cost-competitive</h3>
            <TrustScore value={72} /> {/* Single number */}
        </div>
        
        {/* Scannable: 10-second review */}
        <div className="quick-scan">
            <SourceBar /> {/* Collapsed by default */}
            <KeyPoints />
            <BiasIndicator level="low" /> {/* Only if concerning */}
        </div>
        
        {/* Explorable: Deep dive on demand */}
        <ExpandZone>
            <DetailedAnalysis />
            <SourceBreakdown />
            <MissingPerspectives />
        </ExpandZone>
        </ArgumentCard>
        ```

        **Key Design Principles:**
        - **Glanceability**: Core metric visible in 1-2 seconds
        - **Progressive Disclosure**: Details on demand
        - **Visual Quiet**: Muted colors, small indicators
        - **Contextual Warnings**: Only show when actionable
        - **Responsive Density**: Adapt to screen size

        This approach maintains the argument as the hero while subtly integrating trust signals that build confidence without creating anxiety.

