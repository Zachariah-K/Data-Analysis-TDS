#!/usr/bin/env python3
"""
Data Analyst Agent API
A FastAPI-based service for automated data analysis using LLMs
"""

import os
import json
import base64
import traceback
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import tempfile
import io

# Web framework and file handling
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Data processing and analysis
import pandas as pd
import numpy as np
import duckdb
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
from bs4 import BeautifulSoup

# LLM integration (using OpenAI as example - replace with your preferred LLM)
# from openai import AsyncOpenAI

class AnalysisRequest(BaseModel):
    questions: str
    timeout_seconds: int = 180

class AnalysisResponse(BaseModel):
    results: List[Union[str, float, int, dict]]
    execution_time: float
    
class DataAnalystAgent:
    """Core data analyst agent with LLM integration"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.conn = duckdb.connect()
        # Initialize LLM client here if needed
        # self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    async def process_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Process uploaded files and store in temporary location"""
        file_info = {}
        
        for file in files:
            file_path = self.temp_dir / file.filename
            
            # Save file to temporary location
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Analyze file type and content
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
                file_info[file.filename] = {
                    'type': 'csv',
                    'path': str(file_path),
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'sample': df.head().to_dict()
                }
            elif file.filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                file_info[file.filename] = {
                    'type': 'json',
                    'path': str(file_path),
                    'structure': type(data).__name__,
                    'sample': str(data)[:500] if isinstance(data, (list, dict)) else str(data)
                }
            else:
                # Generic file handling
                file_info[file.filename] = {
                    'type': 'unknown',
                    'path': str(file_path),
                    'size': file_path.stat().st_size
                }
                
        return file_info

    async def web_scraping(self, url: str) -> pd.DataFrame:
        """Scrape data from web sources"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find tables first
            tables = soup.find_all('table')
            if tables:
                # Use pandas to parse HTML tables
                dfs = pd.read_html(response.content)
                if dfs:
                    return dfs[0]  # Return first table found
            
            # If no tables, extract structured data
            # This is a simplified example - real implementation would be more sophisticated
            data = []
            for row in soup.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if cells:
                    data.append(cells)
            
            if data:
                return pd.DataFrame(data[1:], columns=data[0] if data else None)
            
            return pd.DataFrame()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Web scraping failed: {str(e)}")

    def analyze_data(self, df: pd.DataFrame, question: str) -> Any:
        """Perform specific data analysis based on question"""
        
        # Simple pattern matching for common analysis types
        question_lower = question.lower()
        
        if 'correlation' in question_lower:
            # Calculate correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                return corr_matrix.to_dict()
        
        elif 'count' in question_lower or 'how many' in question_lower:
            # Count analysis
            if 'movies' in question_lower and '$2' in question_lower:
                # Example: Count movies over $2B before 2000
                # This would need to be adapted based on actual data structure
                return len(df)
        
        elif 'earliest' in question_lower or 'first' in question_lower:
            # Find earliest/first records
            if 'date' in df.columns:
                return df.loc[df['date'].idxmin()].to_dict()
            elif 'year' in df.columns:
                return df.loc[df['year'].idxmin()].to_dict()
        
        # Default: return basic statistics
        return df.describe().to_dict()

    def create_visualization(self, df: pd.DataFrame, chart_type: str, x_col: str = None, y_col: str = None) -> str:
        """Create visualizations and return as base64 encoded string"""
        
        plt.figure(figsize=(10, 6))
        
        if chart_type.lower() == 'scatterplot':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                plt.scatter(df[x_col], df[y_col], alpha=0.7)
                
                # Add regression line if requested
                if len(df) > 1:
                    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                    p = np.poly1d(z)
                    plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2)
                
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{y_col} vs {x_col}')
        
        elif chart_type.lower() == 'histogram':
            if x_col and x_col in df.columns:
                plt.hist(df[x_col].dropna(), bins=30, alpha=0.7)
                plt.xlabel(x_col)
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {x_col}')
        
        elif chart_type.lower() == 'barplot':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                plt.bar(df[x_col], df[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{y_col} by {x_col}')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        # Encode to base64
        plot_b64 = base64.b64encode(plot_data).decode('utf-8')
        return f"data:image/png;base64,{plot_b64}"

    async def process_questions(self, questions_text: str, file_info: Dict[str, Any]) -> List[Any]:
        """Process analysis questions and return results"""
        
        # Parse questions - assuming they're separated by newlines or numbered
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        results = []
        
        for question in questions:
            try:
                # Check if question involves web scraping
                if 'wikipedia' in question.lower() or 'http' in question.lower():
                    # Extract URL from question
                    words = question.split()
                    url = next((word for word in words if word.startswith('http')), None)
                    
                    if url:
                        df = await self.web_scraping(url)
                        
                        # Process the scraped data based on question
                        if 'scatterplot' in question.lower() or 'scatter plot' in question.lower():
                            # Create scatterplot
                            if len(df.columns) >= 2:
                                viz = self.create_visualization(df, 'scatterplot', df.columns[0], df.columns[1])
                                results.append(viz)
                            else:
                                results.append("Insufficient data for scatterplot")
                        else:
                            # Analyze the data
                            analysis_result = self.analyze_data(df, question)
                            results.append(analysis_result)
                
                # Check if question involves existing file analysis
                elif any(filename in question.lower() for filename in file_info.keys()):
                    # Find the relevant file
                    relevant_file = None
                    for filename, info in file_info.items():
                        if filename.lower() in question.lower() or info['type'] in question.lower():
                            relevant_file = info
                            break
                    
                    if relevant_file and relevant_file['type'] == 'csv':
                        df = pd.read_csv(relevant_file['path'])
                        
                        # Process based on question type
                        if 'plot' in question.lower() or 'chart' in question.lower():
                            # Determine chart type and create visualization
                            if 'scatter' in question.lower():
                                if len(df.columns) >= 2:
                                    viz = self.create_visualization(df, 'scatterplot', df.columns[0], df.columns[1])
                                    results.append(viz)
                                else:
                                    results.append("Insufficient columns for scatterplot")
                            else:
                                results.append("Chart type not specified or supported")
                        else:
                            # Regular analysis
                            analysis_result = self.analyze_data(df, question)
                            results.append(analysis_result)
                
                # For SQL queries on data lakes (DuckDB example)
                elif 'duckdb' in question.lower() or 's3://' in question.lower():
                    # This would handle queries to cloud data sources
                    # Example implementation for the Indian court judgments dataset
                    if 'indian-high-court' in question.lower():
                        # Install and load required extensions
                        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
                        self.conn.execute("INSTALL parquet; LOAD parquet;")
                        
                        # Example query - would need to be adapted based on actual question
                        query = """
                        SELECT COUNT(*) as total_judgments 
                        FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
                        """
                        
                        result = self.conn.execute(query).fetchall()
                        results.append(result[0][0] if result else "No data found")
                
                else:
                    # General question processing - would integrate with LLM here
                    results.append(f"Question processed: {question}")
                    
            except Exception as e:
                results.append(f"Error processing question '{question}': {str(e)}")
        
        return results

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent API",
    description="Automated data analysis service with LLM integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = DataAnalystAgent()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent API is running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/")
async def analyze_data(
    request: Request,
    questions_txt: UploadFile = File(None),
    files: List[UploadFile] = File(default=[])
):
    """Main analysis endpoint"""
    start_time = datetime.now()
    
    try:
        # Extract questions from uploaded file
        if questions_txt:
            questions_content = await questions_txt.read()
            questions_text = questions_content.decode('utf-8')
        else:
            # Try to get from form data
            form_data = await request.form()
            questions_text = form_data.get('questions', '')
        
        if not questions_text:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        # Process additional files
        file_info = {}
        if files and files[0].filename:  # Check if files were actually uploaded
            file_info = await agent.process_files(files)
        
        # Process questions and generate results
        results = await agent.process_questions(questions_text, file_info)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Handle timeout check (implement if needed)
        if execution_time > 180:  # 3 minutes timeout
            raise HTTPException(status_code=408, detail="Request timeout")
        
        return results
        
    except Exception as e:
        # Log error details
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Error in analyze_data: {error_details}")
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    """Alternative analysis endpoint with JSON input"""
    start_time = datetime.now()
    
    try:
        # Process questions without files
        results = await agent.process_questions(request.questions, {})
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            results=results,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "data-analyst-agent:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )