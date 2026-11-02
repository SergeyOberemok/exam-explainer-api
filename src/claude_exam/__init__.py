def main() -> None:
    import uvicorn

    uvicorn.run("claude_exam.api.main:app", host="0.0.0.0", port=8000)
