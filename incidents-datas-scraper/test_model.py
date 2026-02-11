"""
Test the fine-tuned DevOps Incident Responder model.
"""

from unsloth import FastLanguageModel

MODEL_PATH = "./output/merged_model"
MAX_SEQ_LENGTH = 2048

# Test incidents
TEST_INCIDENTS = [
    {
        "name": "Kubernetes OOMKilled",
        "tech": "kubernetes",
        "error": """kubectl describe pod api-server-7d4b8c6f5-x2k9m
State:          Terminated
Reason:         OOMKilled
Exit Code:      137
Restart Count:  5

Events:
  Warning  OOMKilled  Container exceeded memory limit (512Mi)"""
    },
    {
        "name": "Docker connection refused",
        "tech": "docker",
        "error": """docker logs app-container
Error: connect ECONNREFUSED 127.0.0.1:5432
FATAL: Connection to database failed"""
    },
    {
        "name": "Terraform state lock",
        "tech": "terraform",
        "error": """terraform apply
Error: Error acquiring the state lock

Lock Info:
  ID:        abc-123-def
  Path:      s3://my-bucket/terraform.tfstate
  Operation: OperationTypeApply
  Who:       user@host
  Created:   2024-01-15 10:30:00 UTC"""
    },
    {
        "name": "Redis max memory",
        "tech": "redis",
        "error": """redis-cli SET mykey "value"
(error) OOM command not allowed when used memory > 'maxmemory'."""
    },
    {
        "name": "PostgreSQL too many connections",
        "tech": "postgresql",
        "error": """psql -U postgres
FATAL: too many connections for role "postgres"
DETAIL: max_connections is 100 but 100 connections are already open."""
    },
]

SYSTEM_PROMPT = """You are an expert DevOps engineer and SRE. Analyze the provided error logs, stack traces, or incident descriptions.

Your response should include:
1. **Root Cause**: What is causing this issue
2. **Severity**: Low / Medium / High / Critical
3. **Fix**: Step-by-step solution to resolve the issue
4. **Prevention**: How to prevent this in the future (optional)

Be direct, specific, and actionable. Reference exact commands, config changes, or code fixes when applicable."""


def load_model():
    print("Loading model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate_response(model, tokenizer, tech: str, error: str) -> str:
    """Generate incident response."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this {tech} incident and provide diagnosis and fix:\n\n```\n{error}\n```"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response


def main():
    model, tokenizer = load_model()
    
    print("=" * 60)
    print("DEVOPS INCIDENT RESPONDER - TEST")
    print("=" * 60)
    
    for test in TEST_INCIDENTS:
        print(f"\n--- {test['name']} ---")
        print(f"Tech: {test['tech']}")
        print(f"\nError:\n{test['error'][:200]}...")
        
        response = generate_response(model, tokenizer, test['tech'], test['error'])
        
        print(f"\n🤖 Response:\n{response}")
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Paste error logs (end with 'END' on a new line), or 'quit' to exit")
    print("=" * 60)
    
    while True:
        print("\nTech (kubernetes/docker/terraform/etc): ", end="")
        tech = input().strip()
        
        if tech.lower() == 'quit':
            break
        
        print("Paste error logs (end with 'END'):")
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        
        error = '\n'.join(lines)
        
        if error:
            response = generate_response(model, tokenizer, tech, error)
            print(f"\n🤖 Response:\n{response}")


if __name__ == "__main__":
    main()
