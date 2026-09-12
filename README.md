# Awesome Claude Code Agents 🤖

A comprehensive collection of specialized Claude Code agents for software development, designed as **expert consultants** that provide architectural guidance, decision frameworks, and best practices. Each agent focuses on helping you make informed technology choices and design decisions rather than generating code.

## 🚀 Quick Start

### Setting Up Claude Code Agents

1. **Install Claude Code**: Follow the [official installation guide](https://docs.anthropic.com/en/docs/claude-code/installation)

2. **Clone this repository**:
   ```bash
   git clone https://github.com/supatest-ai/awesome-claude-code-agents.git
   cd awesome-claude-code-agents
   ```

3. **Copy agent files**: Copy the agent `.md` files to your Claude Code agents directory:
   ```bash
   cp */*.md ~/.config/claude-code/agents/
   ```

4. **Verify installation**: List available agents:
   ```bash
   claude-code agents list
   ```

### How to Use Agents

Claude Code agents can be invoked in several ways:

1. **Automatic Invocation**: Agents with `PROACTIVELY` in their description will be automatically invoked based on context
2. **Manual Invocation**: Use `@agent-name` in your messages to explicitly invoke an agent
3. **Task-Specific**: Agents will be suggested based on the files you're working with and the tasks you're performing

For more details, see the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code).

## 📁 Repository Structure

```
awesome-claude-code-agents/
├── README.md                          # This file
├── CONTRIBUTING.md                    # Contribution guidelines
├── languages/                         # Programming language experts
│   ├── python-expert.md
│   ├── javascript-typescript-expert.md
│   ├── java-expert.md
│   ├── go-expert.md
│   ├── csharp-expert.md
│   ├── rust-expert.md
│   ├── ruby-expert.md
│   ├── php-expert.md
│   ├── swift-expert.md
│   └── kotlin-expert.md
├── frameworks/                        # Framework specialists
│   ├── react-architect.md
│   ├── vue-specialist.md
│   ├── angular-expert.md
│   ├── django-expert.md
│   └── spring-boot-expert.md
├── development-workflows/             # Development workflow experts
│   └── code-review-master.md
├── architecture/                      # System architecture specialists
│   ├── design-patterns-expert.md
│   ├── microservices-architect.md
│   └── clean-architecture-expert.md
├── testing/                          # Testing and quality assurance
│   ├── test-automation-specialist.md
│   └── performance-testing-expert.md
├── performance/                      # Performance specialists
│   └── performance-optimization-specialist.md
├── data-science/                     # Data science and analytics
│   └── python-data-scientist.md
├── llm-engineering/                 # LLM engineering and operations
│   ├── prompt-engineering-specialist.md
│   ├── rag-architecture-expert.md
│   ├── llmops-engineer.md
│   ├── multi-agent-systems-architect.md
│   ├── llm-finetuning-expert.md
│   └── llm-observability-specialist.md
├── ai-ml/                           # AI and machine learning
│   └── machine-learning-engineer.md
├── devops/                          # DevOps and infrastructure
│   ├── aws-cloud-architect.md
│   ├── docker-specialist.md
│   ├── kubernetes-expert.md
│   └── terraform-infrastructure-expert.md
├── industries/                      # Industry domain experts
│   ├── fintech-security-expert.md
│   └── healthcare-hipaa-expert.md
├── templates/                       # Agent templates for custom agents
│   ├── basic-agent-template.md
│   └── advanced-agent-template.md
└── docs/                           # Additional documentation
    ├── agent-development-guide.md
    ├── best-practices.md
    └── examples/
```

## 🎯 Agent Philosophy

### 🧠 **Guidance-First Approach**
Our agents function as **specialized architectural consultants** rather than code generators. They help you:

- **Make Technology Decisions**: When to use React vs Vue, PostgreSQL vs MongoDB
- **Choose Architecture Patterns**: Microservices vs monolith, CQRS vs traditional CRUD
- **Optimize Performance**: Caching strategies, database tuning, scaling approaches
- **Navigate Ecosystems**: Library selection, version management, best practices
- **Solve Design Problems**: Code organization, testing strategies, deployment patterns

### 📏 **Agent Design Principles**
- **Decision Frameworks**: Clear when/why/how guidance for technology choices
- **Architectural Focus**: System design and structural decisions over syntax
- **Production-Ready**: Enterprise-grade patterns and real-world considerations
- **Ecosystem Navigation**: Modern tooling and library selection guidance
- **Best Practices**: Security, performance, and maintainability built-in

## 🎯 Agent Categories

### 🔤 Languages
Ecosystem guides for programming languages focusing on framework selection, performance optimization, and architectural decisions.

- **[Python Expert](./languages/python-expert.md)** - Ecosystem guidance for Python 3.8+, framework selection (Django/FastAPI/Flask), and performance optimization
- **[JavaScript/TypeScript Expert](./languages/javascript-typescript-expert.md)** - Modern JS/TS tooling decisions, build system selection, and performance patterns
- **[Java Expert](./languages/java-expert.md)** - Java 17+ ecosystem choices, Spring vs alternatives, JVM optimization, and enterprise patterns
- **[Go Expert](./languages/go-expert.md)** - Go project architecture, library selection, concurrency patterns, and performance optimization
- **[Haskell Expert](./languages/haskell-expert.md)** - Advanced functional programming, type system features, STM, and performance optimization
- **[Clojure Expert](./languages/clojure-expert.md)** - Functional programming, immutable data structures, concurrency primitives, and ClojureScript

### 🏗️ Frameworks
Architecture specialists for popular frameworks focusing on project structure, state management decisions, and performance optimization.

- **[React Architect](./frameworks/react-architect.md)** - React 18+ architectural decisions, state management selection, performance optimization, and testing strategies
- **[Vue Specialist](./frameworks/vue-specialist.md)** - Vue 3 Composition API architecture, Pinia vs alternatives, performance patterns, and modern tooling
- **[Angular Expert](./frameworks/angular-expert.md)** - Angular 17+ architectural patterns, state management, RxJS optimization, and enterprise scaling
- **[Django Expert](./frameworks/django-expert.md)** - Django architecture decisions, DRF vs alternatives, async patterns, and production deployment
- **[Spring Boot Expert](./frameworks/spring-boot-expert.md)** - Spring Boot 3+ architectural choices, reactive programming decisions, and microservices patterns

- **[Ivy Tendril](https://github.com/Ivy-Interactive/Ivy-Tendril)** - Open-source agentic software factory with an amazing UI that handles parallel Git worktrees for you, complete with programmatic verifications and fast review loops
### 🔄 Development Workflows
Agents focused on development workflows, code quality, and team collaboration with comprehensive automation and best practices.

#### Development Workflow
- **[Git Workflow Expert](./development-workflows/git-workflow-expert.md)** - Advanced Git strategies, branching models (Git Flow, GitHub Flow), merge conflict resolution, and hooks automation
- **[CI/CD Pipeline Architect](./development-workflows/cicd-pipeline-architect.md)** - GitHub Actions, GitLab CI, Jenkins pipelines, testing automation, and deployment strategies
- **[Project Setup Wizard](./development-workflows/project-setup-wizard.md)** - Multi-language project scaffolding, tooling configuration, and development environment setup
- **[Code Quality Guardian](./development-workflows/code-quality-guardian.md)** - Linting, formatting, pre-commit hooks, quality gates, and technical debt management
- **[Documentation Specialist](./development-workflows/documentation-specialist.md)** - README optimization, API docs, architectural decision records, and documentation automation
- **[Dependency Manager](./development-workflows/dependency-manager.md)** - Package management, security auditing, version updates, and license compliance

#### Team Collaboration
- **[Code Review Master](./development-workflows/code-review-master.md)** - Comprehensive code review focusing on security, performance, maintainability, and best practices
- **[Agile Sprint Planner](./development-workflows/agile-sprint-planner.md)** - User story creation, sprint planning, estimation techniques, and backlog management
- **[Technical Debt Analyst](./development-workflows/technical-debt-analyst.md)** - Technical debt identification, refactoring strategies, and priority assessment

#### Testing & Quality
- **[Test Strategy Architect](./development-workflows/test-strategy-architect.md)** - Testing pyramid, test planning, coverage analysis, and test automation frameworks
- **[Security Audit Expert](./development-workflows/security-audit-expert.md)** - Security scanning, vulnerability assessment, penetration testing, and secure coding practices
- **[Performance Profiler](./development-workflows/performance-profiler.md)** - Performance testing, bottleneck identification, load testing, and optimization strategies

#### Release & Deployment
- **[Release Manager](./development-workflows/release-manager.md)** - Release planning, changelog generation, version management, and rollback strategies
- **[Environment Manager](./development-workflows/environment-manager.md)** - Development, staging, production environments, and infrastructure configuration management

### 🏛️ Architecture
System architecture specialists focusing on pattern selection, design decisions, and scalability strategies.

- **[Design Patterns Expert](./architecture/design-patterns-expert.md)** - Pattern selection guidance, when to use specific patterns, anti-pattern avoidance, and architectural decisions
- **[Microservices Architect](./architecture/microservices-architect.md)** - Microservices vs monolith decisions, service decomposition strategies, and distributed systems patterns
- **[Clean Architecture Expert](./architecture/clean-architecture-expert.md)** - Architectural boundaries, dependency management, and enterprise application design

### 🧪 Testing
Comprehensive testing strategies and quality assurance practices.

- **[Test Automation Specialist](./testing/test-automation-specialist.md)** - TDD/BDD practices, testing pyramid, E2E testing, property-based testing, and CI/CD integration
- **[Performance Testing Expert](./testing/performance-testing-expert.md)** - Load testing, stress testing, performance monitoring, bottleneck identification, and optimization

### ⚡ Performance
Performance specialists focused on optimization and scalability.

- **[Performance Optimization Specialist](./performance/performance-optimization-specialist.md)** - Frontend/backend optimization, database tuning, caching strategies, and monitoring

### 🚀 DevOps
Infrastructure architecture specialists focusing on deployment strategies, scalability decisions, and operational excellence.

- **[Docker Specialist](./devops/docker-specialist.md)** - Containerization strategy, optimization patterns, security considerations, and production deployment decisions
- **[Kubernetes Expert](./devops/kubernetes-expert.md)** - Orchestration architecture, deployment strategies, scaling patterns, and operational excellence
- **[AWS Cloud Architect](./devops/aws-cloud-architect.md)** - Service selection framework, architectural patterns, cost optimization, and Well-Architected principles
- **[Terraform Infrastructure Expert](./devops/terraform-infrastructure-expert.md)** - Infrastructure as Code patterns, state management strategies, and multi-cloud architecture

### 🧠 LLM Engineering
Specialized agents for Large Language Model engineering, operations, and production systems with focus on modern LLM application development.

- **[Prompt Engineering Specialist](./llm-engineering/prompt-engineering-specialist.md)** - Advanced prompt design, optimization techniques, few-shot learning, chain-of-thought reasoning, and evaluation frameworks
- **[RAG Architecture Expert](./llm-engineering/rag-architecture-expert.md)** - Retrieval-Augmented Generation systems, vector databases, document processing, and hybrid search strategies
- **[LLMOps Engineer](./llm-engineering/llmops-engineer.md)** - Production LLM deployment, model serving, scaling strategies, cost optimization, and operational monitoring
- **[Multi-Agent Systems Architect](./llm-engineering/multi-agent-systems-architect.md)** - Complex agent orchestration, coordination patterns, collaborative workflows, and distributed AI systems
- **[LLM Fine-tuning Expert](./llm-engineering/llm-finetuning-expert.md)** - Parameter-efficient fine-tuning (PEFT), LoRA/QLoRA techniques, dataset preparation, and training optimization
- **[LLM Observability Specialist](./llm-engineering/llm-observability-specialist.md)** - Comprehensive monitoring, performance tracking, cost analysis, quality assessment, and system health for LLM applications

### 🤖 Data Science & AI
Data science and machine learning specialists for analytics, modeling, and production ML systems.

- **[Python Data Scientist](./data-science/python-data-scientist.md)** - Data exploration, statistical analysis, feature engineering, and machine learning workflows
- **[Machine Learning Engineer](./ai-ml/machine-learning-engineer.md)** - MLOps, model deployment, monitoring, drift detection, and production ML infrastructure

### 🏢 Industries
Domain experts for specific industries with specialized knowledge and compliance requirements.

- **[FinTech Security Expert](./industries/fintech-security-expert.md)** - Financial services security, PCI DSS compliance, KYC/AML, secure payment processing, and regulatory standards
- **[Healthcare HIPAA Expert](./industries/healthcare-hipaa-expert.md)** - Healthcare technology, HIPAA compliance, medical data security, HL7 FHIR, and patient privacy

## ✨ What Makes These Agents Different

### 🎯 **Decision-Focused Architecture**
Unlike traditional code-generation agents, our agents are designed as **specialized consultants** that help you make informed architectural and technology decisions:

```
Traditional Agents:          Our Agents:
"Here's React code"    →     "When to choose React vs Vue vs Angular"
"FastAPI example"      →     "Django vs FastAPI vs Flask decision matrix"  
"Docker config"        →     "Containerization strategy and patterns"
"AWS Lambda code"      →     "Serverless vs container deployment decisions"
```

### 📊 **Research-Based Design**
Based on analysis of successful Claude Code agent repositories, our agents follow proven patterns:

- **300-800 lines**: Focused, actionable content (vs 2000+ line tutorials)
- **80/20 Guidance/Code**: Decision frameworks over implementations
- **Production-Ready**: Enterprise patterns and real-world considerations  
- **Model-Optimized**: Sonnet for complex decisions, Haiku for simple tasks

### 🏗️ **Architectural Consultant Approach**
Each agent acts as a domain expert consultant providing:

- **Technology Selection**: When/why to choose specific tools and frameworks
- **Architecture Patterns**: System design and structural decisions
- **Performance Optimization**: Scaling strategies and bottleneck identification  
- **Best Practices**: Security, maintainability, and operational excellence
- **Ecosystem Navigation**: Modern tooling and version management

## 🎨 Agent Design Philosophy

Each agent in this collection follows these design principles:

### 🎯 **Focused Expertise**
- **Single Responsibility**: Each agent has a clear, focused area of expertise
- **Deep Knowledge**: Comprehensive understanding of their domain with modern best practices
- **Practical Examples**: Production-ready code examples and real-world scenarios

### 🚀 **Modern Best Practices**
- **Current Technologies**: Up-to-date with latest versions and features
- **Security First**: Security considerations and best practices built-in
- **Performance Oriented**: Optimization patterns and performance considerations

### 📚 **Comprehensive Documentation**
- **Clear Instructions**: Step-by-step guidance and usage examples
- **Code Samples**: Extensive, well-commented code examples
- **Testing Strategies**: Comprehensive testing approaches and examples

### 🔧 **Production Ready**
- **Error Handling**: Robust error handling and edge case management
- **Scalability**: Patterns that work at scale
- **Maintainability**: Clean, readable, and maintainable code practices

## 🛠️ Agent YAML Format

All agents follow the Claude Code subagent format with YAML frontmatter:

```yaml
---
name: agent-name
description: Brief description with PROACTIVELY keyword for auto-invocation
tools: Read, Write, Edit, Bash, Grep, Glob, MultiEdit
---
```

### Required Fields:
- **name**: Unique identifier for the agent (kebab-case)
- **description**: Clear description of agent capabilities and when to use it
- **tools**: List of Claude Code tools the agent can access

### Best Practices:
- Use `PROACTIVELY` in description for agents that should be auto-invoked
- Be specific about when and how the agent should be used
- Include the most relevant tools for the agent's domain

## 🤝 Contributing

We welcome contributions from the community! See our [Contributing Guidelines](./CONTRIBUTING.md) for details on:

- Adding new agents
- Improving existing agents
- Reporting issues
- Suggesting enhancements

### Quick Contribution Steps:

1. Fork the repository
2. Create a new agent using our [templates](./templates/)
3. Follow our [best practices guide](./docs/best-practices.md)
4. Submit a pull request

## 📖 Additional Resources

- **[Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)** - Official Claude Code documentation
- **[Agent Development Guide](./docs/agent-development-guide.md)** - Detailed guide for creating custom agents
- **[Best Practices](./docs/best-practices.md)** - Best practices for agent development
- **[Examples](./docs/examples/)** - Example use cases and implementations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon the excellent work of the open-source community. Special thanks to:

- The Claude Code team at Anthropic
- Contributors to popular frameworks and libraries referenced in our agents
- The broader developer community for sharing knowledge and best practices

## ⭐ Support

If you find this collection helpful, please consider:

- ⭐ Starring the repository
- 🐛 Reporting issues or bugs
- 💡 Suggesting new agents or improvements
- 🤝 Contributing to the project
- 📢 Sharing with other developers

---

**Built with ❤️ for the developer community**