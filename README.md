# Data Analyst Agent

Deploy a data analyst agent. This is an API that uses LLMs to source, prepare, analyze, and visualize any data.

## Project Overview

This project implements a FastAPI-based data analysis service that can:
- Accept file uploads and questions via POST requests
- Perform web scraping for data collection
- Analyze data using pandas, numpy, and statistical methods
- Generate visualizations using matplotlib/seaborn
- Query large datasets using DuckDB
- Return results in various formats including base64-encoded charts

## Features

- **File Processing**: Handles CSV, JSON, and other file formats
- **Web Scraping**: Can extract data from Wikipedia and other web sources
- **Data Analysis**: Performs statistical analysis, correlations, and data mining
- **Visualization**: Creates charts and plots encoded as base64 data URIs
- **Cloud Data**: Supports querying data from S3 buckets using DuckDB
- **Timeout Handling**: Enforces 3-minute execution limits
- **Docker Support**: Fully containerized for easy deployment

## API Endpoints

### Main Analysis Endpoint
```
POST /
```
Accepts multipart form data with:
- `questions.txt`: Text file containing analysis questions (REQUIRED)
- Additional files: CSV, JSON, or other data files (OPTIONAL)

### Alternative JSON Endpoint
```
POST /analyze
```
Accepts JSON body with:
```json
{
    "questions": "Your analysis questions",
    "timeout_seconds": 180
}
```

### Health Check
```
GET /health
```
Returns API status and version information.

## Setup Instructions

### Local Development

1. **Clone and Install Dependencies**
```bash
git clone <your-repo>
cd data-analyst-agent
pip install -r requirements.txt
```

2. **Run the Application**
```bash
python data-analyst-agent.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and Run with Docker**
```bash
docker build -t data-analyst-agent .
docker run -p 8000:8000 data-analyst-agent
```

2. **Using Docker Compose**
```bash
docker-compose up --build
```

### Cloud Deployment Options

#### Heroku
1. Create a Heroku app
2. Add `Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```
3. Deploy via Git or CLI

#### Vercel
1. Add `vercel.json`:
```json
{
    "builds": [{"src": "main.py", "use": "@vercel/python"}],
    "routes": [{"src": "/(.*)", "dest": "main.py"}]
}
```
2. Deploy with Vercel CLI

#### Railway/Render
- Deploy directly from GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Using ngrok for Development
```bash
# Install ngrok and get auth token from ngrok.com
ngrok config add-authtoken <your-token>

# Start your local server
python data-analyst-agent.py

# In another terminal, expose to public
ngrok http 8000
```

## Usage Examples

### Example 1: Wikipedia Data Analysis
```bash
curl -X POST "https://your-api.com/" \
  -F "questions.txt=@questions.txt" \
```

Where `questions.txt` contains:
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line.
```

### Example 2: CSV Data Analysis
```bash
curl -X POST "https://your-api.com/" \
  -F "questions.txt=@analysis_questions.txt" \
  -F "data.csv=@your_data.csv"
```

### Example 3: Cloud Data Analysis (DuckDB + S3)
```bash
curl -X POST "https://your-api.com/" \
  -F "questions.txt=@cloud_questions.txt"
```

Where questions reference S3 data sources like:
```
Query the Indian high court judgement dataset and answer:
1. Which high court disposed the most cases from 2019-2022?
2. What's the regression slope of date_of_registration - decision_date by year?
3. Plot the delay analysis as a scatterplot with regression line.
```

## Architecture

The application follows a modular architecture:

- **DataAnalystAgent**: Core analysis engine
- **FastAPI**: Web framework for API endpoints
- **File Processing**: Handles various file formats
- **Web Scraping**: BeautifulSoup + requests for data collection
- **Data Analysis**: Pandas + NumPy + SciPy for processing
- **Visualization**: Matplotlib + Seaborn for charts
- **Cloud Querying**: DuckDB for large dataset analysis

## Security Considerations

- Input validation on all uploaded files
- Timeout enforcement to prevent long-running processes
- Temporary file cleanup
- CORS configuration for cross-origin requests
- Consider adding authentication for production deployments

## Performance Optimization

- Asynchronous file processing
- Memory-efficient data handling
- Caching for repeated operations
- Resource limits and cleanup
- Docker multi-stage builds for smaller images

## Extending the Agent

To add LLM integration:

1. Install your preferred LLM client (OpenAI, Anthropic, etc.)
2. Add API key configuration
3. Implement question parsing and code generation
4. Add LLM-generated query execution

Example OpenAI integration:
```python
from openai import AsyncOpenAI

class DataAnalystAgent:
    def __init__(self):
        self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def generate_analysis_code(self, question: str, data_info: dict):
        # Use LLM to generate analysis code based on question and data structure
        pass
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all dependencies are installed via requirements.txt
2. **File Upload Issues**: Check multipart form data configuration
3. **Timeout Errors**: Optimize queries or increase timeout limits
4. **Memory Issues**: Monitor data size and implement streaming for large datasets
5. **Docker Build Fails**: Check system dependencies and Docker configuration

### Debug Mode

Run with debug logging:
```bash
PYTHONUNBUFFERED=1 python data-analyst-agent.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Deployment Checklist

- [ ] Test locally with sample data
- [ ] Build and test Docker container
- [ ] Configure environment variables
- [ ] Set up monitoring and logging
- [ ] Configure auto-scaling if needed
- [ ] Test API endpoints with curl/Postman
- [ ] Verify file upload functionality
- [ ] Test timeout and error handling
- [ ] Monitor resource usage
- [ ] Set up continuous deployment

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation
- Submit GitHub issues for bugs
- Contact team for production deployment help