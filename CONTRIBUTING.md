# Contributing to Awesome Claude Code Agents

Thank you for your interest in contributing to this collection! We welcome contributions from developers of all skill levels. This guide will help you get started with contributing new agents, improving existing ones, or enhancing the documentation.

## 🤝 How to Contribute

### Types of Contributions

1. **New Agents**: Create agents for languages, frameworks, or domains not yet covered
2. **Agent Improvements**: Enhance existing agents with better examples, patterns, or features
3. **Bug Fixes**: Fix errors, typos, or incorrect information in existing agents
4. **Documentation**: Improve README, guides, or agent documentation
5. **Templates**: Enhance or create new agent templates
6. **Examples**: Add practical usage examples and case studies

### Before You Start

1. **Check Existing Issues**: Look through [existing issues](https://github.com/your-repo/issues) to see if your contribution is already being worked on
2. **Search Existing Agents**: Make sure your proposed agent doesn't duplicate existing functionality
3. **Review Guidelines**: Read this entire contributing guide and our [best practices](./docs/best-practices.md)

## 📋 Contribution Process

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/awesome-claude-code-sub-agents.git
cd awesome-claude-code-sub-agents

# Add the upstream remote
git remote add upstream https://github.com/supatest-ai/awesome-claude-code-sub-agents.git
```

### 2. Create a Branch

```bash
# Create a new branch for your contribution
git checkout -b feature/agent-name-or-description
```

### 3. Make Your Changes

Follow the guidelines below based on your contribution type.

### 4. Test Your Changes

- Validate YAML frontmatter syntax
- Check Markdown formatting
- Test any code examples
- Ensure all links work correctly

### 5. Submit a Pull Request

```bash
# Commit your changes
git add .
git commit -m "Add [agent-name] agent with [brief description]"

# Push to your fork
git push origin feature/agent-name-or-description

# Create a Pull Request on GitHub
```

## 🤖 Creating New Agents

### Agent Requirements

All agents must meet these minimum requirements:

#### ✅ Required Elements
- **Valid YAML frontmatter** with name, description, and tools
- **Clear expertise area** - specific domain or technology focus
- **Production-ready code examples** - real-world, usable patterns
- **Modern best practices** - current versions and recommended approaches
- **Error handling** - robust error management in examples
- **Testing examples** - appropriate testing strategies for the domain
- **Security considerations** - security best practices where relevant

#### 🎯 Quality Standards
- **Comprehensive coverage** of the domain area
- **Well-commented code** with explanations
- **Multiple complexity levels** - basic to advanced examples
- **Performance considerations** - efficient, scalable patterns
- **Accessibility** - inclusive design practices where applicable

### Agent Structure

Use our templates as starting points:
- **[Basic Agent Template](./templates/basic-agent-template.md)** - For simpler, focused agents
- **[Advanced Agent Template](./templates/advanced-agent-template.md)** - For comprehensive, expert-level agents

### YAML Frontmatter Guidelines

```yaml
---
name: your-agent-name                    # kebab-case, unique identifier
description: Brief description of capabilities. Use PROACTIVELY for auto-invocation.
tools: Read, Write, Edit, Bash, Grep, Glob, MultiEdit
---
```

#### Name Convention
- Use **kebab-case** (lowercase with hyphens)
- Be **descriptive** but **concise**
- Include expertise level if relevant (`expert`, `specialist`, `master`)
- Examples: `python-expert`, `react-architect`, `security-specialist`

#### Description Guidelines  
- **First sentence**: What the agent does
- **Second sentence**: When to use it (include `PROACTIVELY` if auto-invoked)
- **Keep it under 200 characters**
- **Be specific** about capabilities and use cases

#### Tools Selection
Choose tools based on agent needs:
- **Read**: File reading and analysis
- **Write**: Creating new files  
- **Edit**: Modifying existing files
- **MultiEdit**: Multiple file modifications
- **Bash**: Command execution
- **Grep**: Text search and pattern matching
- **Glob**: File pattern matching
- **Task**: Delegating to other agents

### Code Examples Standards

#### Language and Framework Versions
- Use **current stable versions** (not cutting-edge beta)
- Specify version requirements in examples
- Show **modern features** and best practices

#### Code Quality
```typescript
// ✅ Good Example: Well-commented, modern, comprehensive
interface UserService {
  /**
   * Retrieves user data with caching and error handling
   * @param userId - Unique user identifier
   * @returns Promise resolving to user data or null if not found
   */
  getUser(userId: string): Promise<User | null>;
}

class UserServiceImpl implements UserService {
  private cache = new Map<string, User>();
  
  async getUser(userId: string): Promise<User | null> {
    // Check cache first for performance
    if (this.cache.has(userId)) {
      return this.cache.get(userId)!;
    }
    
    try {
      const response = await fetch(`/api/users/${userId}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          return null; // User not found
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const user = await response.json();
      this.cache.set(userId, user); // Cache for future requests
      
      return user;
    } catch (error) {
      console.error(`Failed to fetch user ${userId}:`, error);
      throw error; // Re-throw for upstream handling
    }
  }
}
```

```typescript
// ❌ Bad Example: Minimal, no error handling, outdated patterns
function getUser(id) {
  return fetch('/users/' + id).then(res => res.json());
}
```

## 📁 File Organization

### Directory Structure

Place your agent in the appropriate category:

```
awesome-claude-code-sub-agents/
├── languages/               # Programming languages (Python, JavaScript, etc.)
├── frameworks/              # Frameworks and libraries (React, Django, etc.)
├── development-workflows/   # Development workflows (testing, deployment, etc.)
├── architecture/            # System architecture and patterns
├── testing/                 # Testing and quality assurance  
├── performance/             # Performance and scalability
├── data-science/            # Data science and analytics
├── ai-ml/                   # AI and machine learning
├── devops/                  # DevOps and infrastructure
├── industries/              # Domain-specific (fintech, healthcare, etc.)
└── templates/               # Agent templates and examples
```

### Naming Convention

- **File names**: `agent-name.md` (matching the YAML name field)
- **Agent names**: Descriptive and specific to the domain
- **Consistency**: Follow existing naming patterns in each category

### Category Guidelines

#### Languages
- Focus on a **single programming language**
- Cover **modern language features** and idioms
- Include **ecosystem tools** and best practices
- Examples: `python-expert.md`, `rust-systems-programmer.md`

#### Frameworks  
- Focus on a **specific framework or library**
- Cover **current version** features and patterns
- Include **integration** with related tools
- Examples: `nextjs-architect.md`, `django-rest-expert.md`

#### Development Workflows
- Focus on **development processes** and practices
- Cover **tools and methodologies**
- Include **team collaboration** patterns
- Examples: `ci-cd-specialist.md`, `code-review-expert.md`

#### Data Science & AI/ML
- **Data Science**: Focus on data analysis, visualization, and statistical modeling
- **AI/ML**: Focus on machine learning, deep learning, and AI applications
- Cover **modern tools** and frameworks in each domain
- Examples: `python-data-scientist.md`, `machine-learning-engineer.md`

#### Industries
- Focus on **domain knowledge** and compliance
- Cover **industry standards** and regulations
- Include **specialized patterns** and practices
- Examples: `healthcare-hipaa-expert.md`, `fintech-compliance-specialist.md`

## 🔍 Review Process

### Pull Request Guidelines

#### PR Title Format
```
Add [agent-name] agent for [domain/technology]
```
or
```
Improve [agent-name] agent with [enhancement description]
```

#### PR Description Template
```markdown
## Changes Made
- Brief bullet points of what was added/changed

## Agent Details
- **Name**: agent-name
- **Category**: category-name  
- **Primary Focus**: main expertise area
- **Key Features**: 3-5 main capabilities

## Testing
- [ ] YAML frontmatter validated
- [ ] Markdown formatting checked
- [ ] Code examples tested
- [ ] Links verified

## Checklist
- [ ] Follows contributing guidelines
- [ ] Uses appropriate template
- [ ] Includes comprehensive examples
- [ ] Has proper error handling
- [ ] Includes testing strategies
- [ ] Documentation is clear and complete
```

### Review Criteria

Reviewers will check for:

#### Content Quality
- ✅ **Accuracy**: Technical information is correct and current
- ✅ **Completeness**: Covers the domain comprehensively  
- ✅ **Clarity**: Examples and explanations are clear
- ✅ **Best Practices**: Follows industry standards
- ✅ **Security**: Includes security considerations

#### Code Examples
- ✅ **Functional**: Code examples work as written
- ✅ **Modern**: Uses current language/framework features
- ✅ **Practical**: Real-world, production-ready patterns
- ✅ **Well-commented**: Clear explanations and documentation
- ✅ **Error handling**: Robust error management

#### Structure and Format  
- ✅ **Template compliance**: Follows template structure
- ✅ **YAML validity**: Frontmatter is correctly formatted
- ✅ **Markdown formatting**: Proper syntax and structure
- ✅ **Consistent style**: Matches existing agent patterns

### Review Timeline

- **Initial response**: Within 2-3 business days
- **Detailed review**: Within 1 week for new agents
- **Follow-up responses**: Within 2-3 business days
- **Final approval**: After all review comments are addressed

## 🎯 Best Practices

### Writing Guidelines

#### Documentation Style
- **Clear and concise**: Avoid unnecessary jargon
- **Action-oriented**: Use active voice and imperative mood
- **Consistent terminology**: Use the same terms throughout
- **Progressive complexity**: Start simple, build to advanced

#### Code Style
- **Consistent formatting**: Follow language-specific conventions
- **Meaningful names**: Use descriptive variable and function names
- **Comprehensive comments**: Explain the "why," not just the "what"  
- **Error handling**: Always include proper error management
- **Performance awareness**: Consider scalability and efficiency

### Common Mistakes to Avoid

#### ❌ Don't
- Copy-paste code without understanding or testing
- Use deprecated features or outdated patterns
- Skip error handling in examples
- Write overly complex examples for basic concepts
- Ignore security considerations
- Use hardcoded values without explanation

#### ✅ Do
- Test all code examples before submitting
- Use current, stable versions of technologies
- Include comprehensive error handling
- Progress from simple to complex examples
- Consider security implications
- Use configurable values with clear documentation

## 🛠️ Development Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/awesome-claude-code-sub-agents.git
cd awesome-claude-code-sub-agents

# Validate YAML frontmatter (optional)
npm install -g yaml-lint
yaml-lint */*.md

# Check Markdown formatting (optional)  
npm install -g markdownlint-cli
markdownlint */*.md

# Test Claude Code integration (if you have Claude Code installed)
mkdir -p ~/.claude/agents
find . -mindepth 2 -maxdepth 2 -type f -name '*.md' \
  ! -path './docs/*' ! -path './templates/*' \
  -exec cp {} ~/.claude/agents/ \;
claude
```

Run `/agents` in Claude Code to confirm the agents are available.

### Testing Your Agent

Before submitting:

1. **Syntax Check**: Validate YAML and Markdown formatting
2. **Code Testing**: Verify all code examples work as written
3. **Link Verification**: Ensure all external links are functional
4. **Claude Code Testing**: Test agent integration if possible

## 💬 Getting Help

### Questions and Discussions

- **General Questions**: Create a [Discussion](https://github.com/repo/discussions)
- **Bug Reports**: Create an [Issue](https://github.com/repo/issues)
- **Feature Requests**: Create an [Issue](https://github.com/repo/issues) with the feature label
- **Real-time Help**: Join our [Discord/Slack community](link)

### Resources

- **[Agent Development Guide](./docs/agent-development-guide.md)** - Detailed development guide
- **[Best Practices](./docs/best-practices.md)** - Coding and documentation standards
- **[Claude Code Documentation](https://code.claude.com/docs/en/sub-agents)** - Official Claude Code sub-agent docs
- **[Examples](./docs/examples/)** - Real-world usage examples

## 🏆 Recognition

### Contributors

All contributors are recognized in our README and releases. Significant contributions may be highlighted in:

- **README.md**: Contributor recognition section
- **Release notes**: Major contributions highlighted
- **Special recognition**: Outstanding contributions featured

### Contribution Types

- 🐛 **Bug fixes**
- 📝 **Documentation**  
- 💡 **New agents**
- ⚡ **Performance improvements**
- 🎨 **Code improvements**
- 🔒 **Security enhancements**

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 📞 Contact

- **Maintainers**: [List of maintainer contacts]
- **Email**: [contact email if available]
- **Issues**: [GitHub issues link]
- **Discussions**: [GitHub discussions link]

---

Thank you for contributing to the Awesome Claude Code Agents collection! Your contributions help make development more efficient and enjoyable for the entire community. 🎉
