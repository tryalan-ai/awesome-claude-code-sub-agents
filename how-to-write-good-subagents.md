# How to Write Good Claude Code Sub-Agents

*Simple principles for creating effective specialized AI assistants*

## The 5 Core Principles (From Anthropic)

1. **Start with Claude-generated agents** - Generate initial subagent with Claude, then customize
2. **Design focused subagents** - Single, clear responsibilities only  
3. **Write detailed prompts** - Include specific instructions, examples, constraints
4. **Limit tool access** - Only grant necessary tools
5. **Version control** - Check project subagents into git for team collaboration

## Basic Structure

```yaml
---
name: your-agent-name
description: When this subagent should be invoked
tools: [optional-tool-list]
---

Your system prompt goes here.
Define role, capabilities, and approach clearly.
Include specific instructions and constraints.
```

## Writing Effective System Prompts

### Be Specific and Detailed
```
❌ "You are a code reviewer"
✅ "You are a senior code reviewer specializing in Python. 
   Focus on security vulnerabilities, performance issues, 
   and PEP 8 compliance. Always suggest specific fixes."
```

### Single Responsibility 
```
❌ "Handle frontend and backend and testing"
✅ "Specialized React component reviewer focused on 
   accessibility, performance, and reusability"
```

### Include Context and Examples
```
✅ "When reviewing API endpoints, check for:
   - Proper error handling (try/catch blocks)
   - Input validation and sanitization  
   - Rate limiting considerations
   - Consistent response formats"
```

## Description Field Best Practices

### Make It Delegation-Friendly
```yaml
# Help Claude know when to use this agent
description: "Use proactively for all React component reviews and optimization"
description: "MUST BE USED when analyzing database performance issues"
description: "Expert in Node.js API security - invoke for backend code review"
```

## Common Templates

### Code Reviewer
```yaml
---
name: code-reviewer
description: Use proactively for comprehensive code analysis
tools: [read, edit]
---

You are a senior software engineer with expertise in code review.
Focus on: security, performance, maintainability, and best practices.
Always provide specific, actionable feedback with code examples.
```

### Test Runner
```yaml
---
name: test-runner  
description: Use to run tests and fix failures automatically
---

You are a test automation expert. When you see code changes:
1. Run appropriate tests proactively
2. If tests fail, analyze root cause
3. Fix failures while preserving test intent
4. Report results clearly
```

### Performance Optimizer
```yaml
---
name: performance-optimizer
description: Use for database queries and system performance analysis
---

You are a performance optimization specialist.
Analyze bottlenecks, suggest specific improvements.
Focus on: query optimization, caching strategies, resource usage.
Measure before/after performance impact.
```

## Key Writing Tips

### Context Is Everything
> "When you're working with a new grad with amnesia, write out all the context you have in your head. The more context you give Claude, the more effective it'll be."

### Remember Patterns
> "If you remember 'there's a similar pattern we've used for this type of problem in this codebase', write it down!"

### Start Small
> "Don't build 20-30 agents at once. Notice when you're repeating a task, and create an agent in that moment."

## What Makes Agents Actually Get Used

### Proactive Language
- "Use proactively"
- "MUST BE USED"  
- "Automatically invoke when..."

### Clear Boundaries
- One specific domain expertise
- Obvious trigger conditions
- Well-defined scope

### Practical Value
- Solves real recurring problems
- Faster than doing manually
- Preserves context and reduces switching

## Simple Examples That Work

```yaml
---
name: security-auditor
description: Use for any security-related code analysis
---

Security specialist focused on OWASP vulnerabilities.
Scan for: SQL injection, XSS, auth bypasses, data exposure.
Provide specific remediation steps.
```

```yaml
---
name: api-designer
description: Use when designing or reviewing REST APIs
---

API design expert following REST principles.
Focus on: resource naming, HTTP methods, status codes,
error handling, documentation, versioning strategies.
```

```yaml  
---
name: database-architect
description: Use for database schema and query optimization
---

Database specialist for schema design and optimization.
Analyze: normalization, indexing, query performance,
migration strategies, data integrity constraints.
```

## The Golden Rule

**Write prompts like you're onboarding a smart junior developer who needs explicit context about your codebase, patterns, and preferences.**

---

## Further Reading and Resources

### Official Documentation
- [Anthropic Claude Code Sub-Agents](https://code.claude.com/docs/en/sub-agents) - Official documentation and configuration guide
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices) - Anthropic's recommended workflows and patterns
- [How Anthropic Teams Use Claude Code](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code) - Real-world usage examples from Anthropic

### Community Resources and Examples
- [Production-Ready Subagents Collection](https://github.com/VoltAgent/awesome-claude-code-subagents) - 100+ specialized agents for full-stack development
- [wshobson/agents](https://github.com/wshobson/agents) - Collection of production-ready subagents for Claude Code
- [17 Claude Code SubAgent Examples](https://medium.com/@joe.njenga/17-claude-code-subagents-examples-with-templates-you-can-use-immediately-c70ef5567308) - Templates you can use immediately

### Practical Guides and Tutorials
- [Claude Code Camp: Workflows Turning One Engineer Into Ten](https://every.to/source-code/claude-code-camp) - Comprehensive workflow guide
- [How to Use Claude Code Subagents for Complex Development Projects](https://goatreview.com/how-to-use-claude-code-subagents-tutorial/) - Step-by-step tutorial
- [Why Every Developer Needs Claude Code Sub Agents](https://medium.com/vibe-coding/why-every-developer-needs-claude-code-sub-agents-and-how-i-build-them-551c2ae4aab0) - Building and implementation guide

### Advanced Topics
- [Claude Code: Subagent Deep Dive](https://cuong.io/blog/2025/06/24-claude-code-subagent-deep-dive) - Technical architecture and advanced usage
- [How I Use Claude Code](https://spiess.dev/blog/how-i-use-claude-code) - Personal workflow and optimization tips
- [Claude Code Subagents Enable Modular AI Workflows](https://www.infoq.com/news/2025/08/claude-code-subagents/) - Technical analysis and industry perspective

*Compiled from official Anthropic documentation and community best practices*
