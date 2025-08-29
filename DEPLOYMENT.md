# Deployment Strategy for Atlan Support Agent v2

## Current Challenge with Vercel

### Why Vercel Won't Work for This Application
1. **Size Limitations**: Vercel has a 250MB unzipped limit for serverless functions
2. **Heavy Dependencies**: Our stack includes:
   - LangChain (30-50MB)
   - Pinecone client (10-20MB)
   - OpenAI/Anthropic SDKs (20-30MB)
   - FastAPI + dependencies (50-70MB)
   - Total: Likely exceeds 250MB limit

3. **Platform Mismatch**: Vercel is optimized for frontend/edge functions, not backend AI applications

## Recommended Deployment Options

### Option 1: Railway (RECOMMENDED)
**Best for:** Quick deployment with minimal configuration

```bash
# Deploy with Railway CLI
railway login
railway link
railway up
```

**Pros:**
- No size limits for containers
- Built-in environment variable management
- Automatic HTTPS
- Simple GitHub integration
- $5/month usage-based pricing

**Setup:**
1. Create account at railway.app
2. Connect GitHub repo
3. Add environment variables
4. Deploy

### Option 2: Render
**Best for:** Production deployments with more control

**Pros:**
- Free tier available
- Supports Docker deployments
- Built-in PostgreSQL if needed
- Good for full-stack apps

**Setup:**
1. Create `render.yaml` in root
2. Connect GitHub
3. Configure environment variables
4. Deploy

### Option 3: Google Cloud Run
**Best for:** Enterprise deployments with scalability

```dockerfile
# Dockerfile for Cloud Run
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Deploy:**
```bash
gcloud run deploy atlan-support-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 4: AWS Lambda (with limitations)
**Only if:** You can optimize dependencies under 250MB

Use AWS SAM or Serverless Framework for deployment.

## Quick Fix for Vercel (If You Must Use It)

### Approach 1: Split Services
1. Deploy FastAPI on Railway/Render
2. Keep frontend on Vercel
3. Use API Gateway pattern

### Approach 2: Minimize Dependencies
Create `requirements-minimal.txt`:
```txt
fastapi==0.104.1
httpx==0.25.2
pydantic>=2.0,<3.0
python-dotenv==1.0.0
```

Then create lightweight endpoints that call external services.

## Immediate Action Plan

1. **For Testing:** Use the minimal test deployment
2. **For Production:** Deploy on Railway (fastest option)
3. **For Enterprise:** Use Google Cloud Run

## Environment Variables Required
Regardless of platform, ensure these are set:
```env
OPENROUTER_API_KEY=xxx
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=atlan-docs
OPENAI_API_KEY=xxx
GENERATION_MODEL=anthropic/claude-sonnet-4
ROUTING_MODEL=anthropic/claude-sonnet-4
EVAL_MODEL=openai/gpt-5-mini
```

## Deployment Commands

### Railway (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway link
railway up
```

### Docker (for any platform)
```bash
# Build
docker build -t atlan-support-agent .

# Run locally
docker run -p 8000:8000 --env-file .env atlan-support-agent

# Deploy to Cloud Run
gcloud run deploy --source .
```

## Conclusion
Given your tech stack with LangChain, Pinecone, and other AI dependencies, **Railway or Render are the best options**. Vercel is not suitable for this type of backend-heavy application.

Would you like to proceed with Railway deployment instead?