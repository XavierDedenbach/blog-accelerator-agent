# Researcher Agent Architecture

## System Overview

The Researcher Agent is a sophisticated component of the Blog Accelerator that analyzes blog content and generates comprehensive research to enhance the quality and credibility of the blog. Below are diagrams illustrating its architecture and workflow.

```mermaid
graph TD
    subgraph Blog Accelerator System
        A[User Proposes Opinionated Concept] --> B[Researcher Agent]
        B --> C[Research Report Stored in DB]
        C --> D[User Writes Blog Post]
        D --> E[Review Agent]
        E --> F[User Reviews Feedback]
        F --> G[User Releases Blog]
    end
    
    subgraph ResearcherAgent Process
        B --> H[Topic Validation & Planning]
        H --> I1[Industry/System Analysis]
        H --> I2[Proposed Solution Analysis]
        H --> I3[Current Paradigm Analysis]
        H --> I4[Audience Analysis]
        I1 & I2 & I3 --> J[Visual Asset Collection]
        I4 --> K[Analogy Generation]
        J & K --> L[Research Integration]
        L --> M[Calculate Readiness Score]
        M --> N[Generate Final Markdown Report]
        N --> O[Store Report in MongoDB]
        O --> P[Open Report in Browser]
    end
```

## Research Methodology

The Researcher Agent implements **sequential thinking** to achieve deeper analysis with more nuance. This methodical approach ensures thorough consideration of all factors before drawing conclusions.

```mermaid
graph TD
    A[Research Topic] --> B[Step 1: Identify Core Constraints]
    B --> C[Step 2: Consider Systemic Context]
    C --> D[Step 3: Map Stakeholder Perspectives]
    D --> E[Step 4: Identify Challenges/Solutions]
    E --> F[Step 5: Generate Supporting Evidence]
    F --> G[Step 6: Test Counter-Arguments]
```

This sequential approach is applied to each research component (Industry, Solution, Paradigm) to ensure all analysis is contextually rich and considers multiple perspectives before reaching conclusions.

## Research Gathering Architecture

```mermaid
graph LR
    subgraph Gather Research Process
        A[Main Topic] --> B[Topic Validation]
        B --> C[Industry Analysis]
        B --> D[Solution Analysis]
        B --> E[Paradigm Analysis]
        B --> F[Audience Analysis]
        C & D & E --> G[Visual Asset Collection]
        F --> H[Analogy Generation]
    end
    
    subgraph Industry Analyzer
        C --> C0[Sequential Reflection on Constraints]
        C0 --> C1[Identify Challenges 10+]
        C1 --> C2[Find Sources for Challenges]
        C2 --> C3[Analyze Challenge Components]
        C3 --> C4[Risk/Slowdown/Cost/Inefficiency Factors]
    end
    
    subgraph Solution Analyzer
        D --> D0[Sequential Reflection on Constraints]
        D0 --> D1[Identify 5-10 Pro Arguments]
        D0 --> D2[Identify 5-10 Counter Arguments]
        D0 --> D3[Find Progress Metrics]
        D0 --> D4[Collect 50-100 Visual Assets]
        D1 --> D5[Find Supporting Sources]
        D2 --> D6[Find Supporting Sources]
    end
    
    subgraph Paradigm Analyzer
        E --> E0[Sequential Reflection on Constraints]
        E0 --> E1[Analyze Current Paradigm]
        E0 --> E2[Research Origin & Previous Paradigm]
        E0 --> E3[Collect 10-20 Visual Representations]
        E0 --> E4[Identify 2-3 Alternative Solutions]
    end
    
    subgraph Audience Analyzer
        F --> F1[Identify Knowledge Gaps]
        F --> F2[Create Acronym Glossary]
        F --> F3[Highlight Difficult Concepts]
        F --> F4[Generate Challenge Analogies]
        F --> F5[Generate Solution Analogies]
    end
```

## Key Components

### 1. ResearcherAgent
Core class that orchestrates the research process. It initializes with various API keys and components:
- Source Validator: Validates and finds sources for claims
- FirecrawlClient: Client for web search and crawling
- Various analyzers: For industry, solution, paradigm, audience analysis, and analogy generation

### 2. Industry Analyzer
Identifies challenges facing industries or systems related to the blog topic:
- **Employs sequential thinking** to first understand core constraints and nuances before identifying challenges
- Reflects on team size, resource limitations, and contextual factors
- Identifies critical challenges (minimum 10)
- Finds supporting sources for each challenge
- Analyzes components of each challenge (risk factors, slowdown factors, cost factors, inefficiency factors)

### 3. Solution Analyzer
Analyzes proposed solutions discussed in the blog:
- **Employs sequential thinking** to understand solution constraints before evaluating effectiveness
- Considers implementation complexity, adoption barriers, and ecosystem dependencies
- Identifies 5-10 pro arguments supporting the solution
- Identifies 5-10 counter arguments against the solution
- Finds key data showing progress with 5-10 metrics
- Collects 50-100 visual assets representing the solution
- Finds supporting sources for all arguments

### 4. Paradigm Analyzer
Analyzes current and historical paradigms related to the topic:
- **Employs sequential thinking** to understand paradigm evolution before assessing effectiveness
- Considers historical context, driving forces, and shifting constraints
- Identifies 5-10 reasons the current paradigm addresses challenges
- Determines when this paradigm was created
- Researches the previous paradigm 
- Collects 10-20 visual representations
- Identifies 2-3 other emerging solutions and their shortcomings

### 5. Audience Analyzer
Analyzes the target audience for the blog:
- Identifies knowledge gaps between research and general audience
- Creates glossary of acronyms with explanations
- Highlights concepts difficult for the audience to understand
- Generates 3 analogies to explain challenges
- Generates 3 analogies to explain proposed solutions

### 6. Visual Asset Collection
Gathers visual materials to support the blog:
- Collects images, videos, infographics related to all aspects
- Stores assets in organized folders by topic
- Ensures variety of visual material types

## Sequential Thinking Implementation

The agent implements sequential thinking through:

1. **Prompt Engineering**: Each analysis prompt requires the LLM to first consider constraints and context before generating output
2. **Multi-Step Reasoning**: Breaking down complex analyses into sequential steps
3. **Constraint Mapping**: Explicitly identifying limitations before evaluating solutions
4. **Perspective Shifting**: Analyzing topics from multiple stakeholder viewpoints sequentially
5. **Chain-of-Thought**: Using structured reasoning to build nuanced understanding

This approach ensures that research output has greater depth and considers more nuance than a single-pass analysis.

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant RunnerScript
    participant ResearcherAgent
    participant IndustryAnalyzer
    participant SolutionAnalyzer
    participant ParadigmAnalyzer
    participant AudienceAnalyzer
    participant VisualAssetCollector
    participant AnalogyGenerator
    participant MongoDB
    participant WebBrowser
    
    User->>RunnerScript: Run researcher agent for topic
    RunnerScript->>ResearcherAgent: Process topic
    ResearcherAgent->>ResearcherAgent: Validate topic requirements
    ResearcherAgent->>ResearcherAgent: Break down into subtopics
    
    par Parallel Research Tasks
        ResearcherAgent->>IndustryAnalyzer: Analyze industry challenges
        Note over IndustryAnalyzer: Apply sequential thinking
        ResearcherAgent->>SolutionAnalyzer: Analyze proposed solution
        Note over SolutionAnalyzer: Apply sequential thinking
        ResearcherAgent->>ParadigmAnalyzer: Analyze current paradigm
        Note over ParadigmAnalyzer: Apply sequential thinking
        ResearcherAgent->>AudienceAnalyzer: Analyze audience needs
    end
    
    IndustryAnalyzer-->>ResearcherAgent: Return challenges analysis
    SolutionAnalyzer-->>ResearcherAgent: Return solution analysis
    ParadigmAnalyzer-->>ResearcherAgent: Return paradigm analysis
    AudienceAnalyzer-->>ResearcherAgent: Return audience analysis
    
    ResearcherAgent->>VisualAssetCollector: Collect visual assets
    ResearcherAgent->>AnalogyGenerator: Generate analogies
    
    VisualAssetCollector-->>ResearcherAgent: Return visual assets
    AnalogyGenerator-->>ResearcherAgent: Return analogies
    
    ResearcherAgent->>ResearcherAgent: Calculate readiness score
    ResearcherAgent->>ResearcherAgent: Generate research report
    ResearcherAgent->>MongoDB: Save research results (report + data)
    
    ResearcherAgent-->>RunnerScript: Return result summary (including report URL)
    RunnerScript->>WebBrowser: Open report URL
    WebBrowser-->>User: Display Research Report
```

## Readiness Score Calculation

The readiness score (A-F) is calculated based on:
- Completeness of industry challenge analysis (10+ challenges)
- Quality of solution arguments (5-10 pro and counter arguments)
- Thoroughness of paradigm analysis
- Depth of audience analysis
- Number of visual assets (50-100 for solution, 10-20 for paradigm)
- Number of analogies (3 for challenges, 3 for solutions)
- Quality and authority of sources

A higher score indicates a more thorough research foundation for the blog post. 