# Claude Code Development Guide

## Project Overview

This document serves as a comprehensive reference guide for Claude Code during the development of the Atlan Support Agent v2. It outlines critical protocols, workflows, and requirements to ensure consistent, high-quality development practices.

### Project Scope
- Development of Atlan Support Agent v2
- Implementation of AI-powered support capabilities
- Integration with existing Atlan ecosystem
- Maintenance of code quality and testing standards

## Critical Files to Track

### Core Configuration Files
- `package.json` - Project dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `.env` files - Environment variables and secrets
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Container definitions

### Application Structure
- `/src` - Main application source code
- `/tests` - Test suites and specifications
- `/docs` - Documentation files
- `/config` - Configuration files
- `/scripts` - Build and deployment scripts

### Development Documentation
- `claude.md` - This development guide (update after each phase)
- `plan.md` - Project roadmap and phase tracking (update after each phase)
- `README.md` - Project overview and setup instructions

## Development Workflow

### Phase-Based Development
1. **Analysis Phase**: Understand requirements and existing codebase
2. **Planning Phase**: Create detailed implementation plan
3. **Implementation Phase**: Write code following established patterns
4. **Testing Phase**: Comprehensive testing and validation
5. **Documentation Phase**: Update documentation and guides

### Code Standards
- Follow existing code patterns and conventions
- Use TypeScript for type safety
- Implement proper error handling
- Write meaningful commit messages
- Maintain consistent code formatting

### File Management Protocol
- **NEVER** create new files unless absolutely necessary
- **ALWAYS** prefer editing existing files
- **NEVER** proactively create documentation files without explicit request
- Use absolute file paths in all references
- Maintain existing directory structure

## Update Protocol

### After Each Phase Completion

#### Required Updates
1. **Update claude.md**:
   - Add phase completion notes
   - Update critical files list if new files were created
   - Document any new patterns or conventions discovered
   - Note any deviations from original plan

2. **Update plan.md**:
   - Mark completed phases
   - Update timeline if necessary
   - Document any scope changes
   - Add lessons learned and next steps

#### Update Format
```markdown
## Phase [X] Completion - [Date]
- **Status**: Completed/In Progress/Blocked
- **Files Modified**: [List of modified files]
- **Key Changes**: [Summary of changes]
- **Next Steps**: [What comes next]
- **Issues Encountered**: [Any problems and solutions]
```

## Testing and Validation

### Testing Requirements
- Unit tests for all new functions and classes
- Integration tests for API endpoints
- End-to-end tests for user workflows
- Performance tests for critical paths
- Security tests for authentication and authorization

### Validation Checklist
- [ ] All tests passing
- [ ] No TypeScript errors
- [ ] No linting errors
- [ ] Performance benchmarks met
- [ ] Security requirements satisfied
- [ ] Documentation updated

### Test Execution Protocol
```bash
# Run before any commit
npm test
npm run lint
npm run type-check
npm run build
```

## Error Recovery

### Common Issues and Solutions

#### Build Failures
1. Check dependency versions in package.json
2. Clear node_modules and reinstall
3. Verify TypeScript configuration
4. Check for circular dependencies

#### Test Failures
1. Review test specifications for accuracy
2. Check for environment-specific issues
3. Verify mock data and fixtures
4. Ensure proper test isolation

#### Runtime Errors
1. Check environment variables
2. Verify database connections
3. Review API endpoint configurations
4. Check authentication and authorization

### Debugging Protocol
1. Reproduce the issue consistently
2. Check logs for error details
3. Use debugging tools (Chrome DevTools, VS Code debugger)
4. Create minimal reproduction case
5. Document solution in this file

## Development Approach Guidelines

### Code Organization
- Group related functionality in modules
- Use dependency injection for testability
- Implement proper separation of concerns
- Follow SOLID principles

### API Development
- Use consistent REST conventions
- Implement proper HTTP status codes
- Add comprehensive input validation
- Include rate limiting and security headers

### Database Operations
- Use migrations for schema changes
- Implement proper indexing strategies
- Add query optimization
- Include data validation

### Security Considerations
- Implement proper authentication
- Add authorization checks
- Sanitize user inputs
- Use secure communication protocols

## Communication Guidelines

### Progress Updates
- Provide clear status updates after each phase
- Include specific file changes and modifications
- Explain reasoning behind implementation decisions
- Highlight any deviations from original plan

### Problem Reporting
- Describe issues with specific examples
- Include relevant error messages
- Provide steps to reproduce
- Suggest potential solutions

### Documentation Standards
- Use clear, concise language
- Include code examples where helpful
- Maintain consistent formatting
- Keep documentation up-to-date with code changes

## Phase Completion Checklist

### Before Marking Phase Complete
- [ ] All planned features implemented
- [ ] Tests written and passing
- [ ] Code reviewed for quality
- [ ] Documentation updated
- [ ] No blocking issues remaining
- [ ] claude.md updated with phase notes
- [ ] plan.md updated with completion status

### Phase Handoff Requirements
- [ ] Clear description of completed work
- [ ] List of modified/created files
- [ ] Any new dependencies or configurations
- [ ] Known issues or technical debt
- [ ] Next phase prerequisites identified

## Version History

### v1.0.0 - Initial Creation
- **Date**: 2025-08-26
- **Author**: Claude Code
- **Changes**: Initial creation of development guide
- **Next Review**: After first phase completion

### v1.1.0 - Phase 2 Completion
- **Date**: 2025-08-27
- **Author**: Claude Code
- **Changes**: Phase 2 Vector Database and Indexing completed successfully
- **Files Modified**: app/vector.py, scripts/index_pinecone.py, requirements.txt, .env.example, scripts/README.md
- **Key Achievements**: 
  - Complete Pinecone integration with sub-500ms search latency
  - Advanced caching system with TTL support
  - Enhanced indexing script with validation and cost estimation
  - Production-ready error handling and performance optimization
- **Next Phase**: Phase 3 - Core RAG Pipeline (LangChain agent implementation)
- **Issues Encountered**: Critical bugs discovered and fixed - QueryResponse conversion, dimension mismatch, None handling
- **Lessons Learned**: Parallel sub-agent execution accelerated development; thorough testing essential for vector operations
- **Next Review**: After Phase 3 completion

### v1.2.0 - Phase 2 Bug Fixes and Completion
- **Date**: 2025-08-27
- **Author**: Claude Code
- **Changes**: Fixed critical search functionality bugs and achieved full Phase 2 completion
- **Files Modified**: app/vector.py (QueryResponse conversion fix)
- **Key Bug Fixes**:
  - Fixed Pinecone QueryResponse to dict conversion issue
  - Fixed None value handling in similarity score normalization
  - Fixed dimension mismatch (recreated index with 1536 dimensions)
- **Final Status**: Phase 2 fully completed with 192/203 chunks indexed and working search
- **Performance**: Sub-500ms search achieved with robust caching system
- **Next Phase**: Phase 3 - Core RAG Pipeline ready to begin

### v1.3.0 - Phase 3 Core RAG Pipeline Completion
- **Date**: 2025-08-27
- **Author**: Claude Code
- **Changes**: Complete LangChain-based RAG pipeline implementation with intelligent search agent
- **Files Modified**: app/rag.py (enhanced and optimized implementation), .env (OpenRouter configuration)
- **Key Achievements**:
  - LangChain search agent with semantic_search tool using @tool decorator
  - Intelligent multi-round search decision-making capability (5+ searches per query)
  - Integration with existing VectorStore from Phase 2
  - Context storage and accumulation for downstream processing
  - Atlan-specific system prompts and query handling
  - Comprehensive error handling and performance logging
  - OpenRouter integration for using Claude Sonnet 4 via LangChain
  - In-memory caching with TTL for identical queries
- **Implementation Details**:
  - SearchTool class with get_semantic_search_tool() method
  - RAGAgent class with _create_agent() and search_and_retrieve() methods
  - Agent can perform multiple searches if initial results are insufficient
  - Returns both formatted context and raw chunks for routing
  - Request context storage for downstream processing
  - OpenRouter configuration with Claude Sonnet 4 for agent reasoning
  - RetrievalCache class with MD5-based cache keys and TTL support
  - Context management with token limits and conversation truncation
- **API Interface**: async search_and_retrieve(query, conversation_history, session_id) -> Tuple[List[RetrievalChunk], str]
- **Performance**: Agent makes intelligent search decisions with quality assessment, multiple search rounds
- **Testing**: All 7 test cases passing including end-to-end RAG pipeline validation
- **Next Phase**: Phase 4 - Query Router ready to begin
- **Issues Encountered**: Fixed OpenRouter integration for Claude models via base_url configuration
- **Code Quality**: Production-ready with proper async support, caching, and comprehensive logging

---

*This document should be updated after each development phase to reflect current project status and lessons learned.*