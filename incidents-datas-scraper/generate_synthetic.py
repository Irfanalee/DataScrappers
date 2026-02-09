"""
Synthetic Data Generator for DevOps Incident Responder
Generates error scenarios + expert diagnoses using Claude Haiku 3.5

Estimated cost: ~$1.50-2.00 for 1,500 examples
"""

import anthropic
import json
import random
import time
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "data/synthetic"
OUTPUT_FILE = f"{OUTPUT_DIR}/synthetic_incidents.json"
TARGET_EXAMPLES = 1500
MODEL = "claude-3-5-haiku-20241022"

# Rate limiting
REQUESTS_PER_MINUTE = 50
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# =============================================================================
# INCIDENT TEMPLATES BY TECHNOLOGY
# =============================================================================

INCIDENT_TEMPLATES = {
    "kubernetes": [
        {
            "scenario": "Pod CrashLoopBackOff",
            "error": """kubectl get pods
NAME                        READY   STATUS             RESTARTS   AGE
api-server-7d4b8c6f5-x2k9m  0/1     CrashLoopBackOff   5          10m

kubectl logs api-server-7d4b8c6f5-x2k9m
Error: Cannot find module '/app/server.js'""",
            "category": "pod_failure"
        },
        {
            "scenario": "OOMKilled container",
            "error": """kubectl describe pod worker-5f7b8d9c4-abc12
State:          Terminated
Reason:         OOMKilled
Exit Code:      137
Last State:     Terminated
Reason:         OOMKilled
Restart Count:  8

Events:
  Warning  OOMKilled  Container exceeded memory limit (512Mi)""",
            "category": "resource"
        },
        {
            "scenario": "ImagePullBackOff",
            "error": """kubectl get pods
NAME                    READY   STATUS             RESTARTS   AGE
web-app-6c7d8e9f0-xyz   0/1     ImagePullBackOff   0          5m

kubectl describe pod web-app-6c7d8e9f0-xyz
Events:
  Warning  Failed   Failed to pull image "myregistry.io/app:v2.1": rpc error: code = Unknown desc = Error response from daemon: unauthorized: authentication required""",
            "category": "image"
        },
        {
            "scenario": "Service not reachable",
            "error": """kubectl exec -it debug-pod -- curl http://backend-service:8080
curl: (7) Failed to connect to backend-service port 8080: Connection refused

kubectl get endpoints backend-service
NAME              ENDPOINTS   AGE
backend-service   <none>      15m""",
            "category": "networking"
        },
        {
            "scenario": "PersistentVolumeClaim pending",
            "error": """kubectl get pvc
NAME        STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   AGE
data-pvc    Pending                                      standard       10m

kubectl describe pvc data-pvc
Events:
  Warning  ProvisioningFailed  storageclass.storage.k8s.io "standard" not found""",
            "category": "storage"
        },
        {
            "scenario": "RBAC permission denied",
            "error": """kubectl get pods -n production
Error from server (Forbidden): pods is forbidden: User "developer@company.com" cannot list resource "pods" in API group "" in the namespace "production" """,
            "category": "rbac"
        },
        {
            "scenario": "Node NotReady",
            "error": """kubectl get nodes
NAME           STATUS     ROLES    AGE   VERSION
worker-node-1  NotReady   <none>   30d   v1.28.0

kubectl describe node worker-node-1
Conditions:
  Type             Status  Reason
  MemoryPressure   True    KubeletHasInsufficientMemory
  DiskPressure     True    KubeletHasDiskPressure""",
            "category": "node"
        },
        {
            "scenario": "Ingress 502 Bad Gateway",
            "error": """curl -I https://app.example.com
HTTP/2 502
server: nginx

kubectl logs -n ingress-nginx ingress-nginx-controller-xxx
upstream connect error or disconnect/reset before headers. reset reason: connection termination""",
            "category": "ingress"
        },
    ],
    "docker": [
        {
            "scenario": "Container build fails - dependency",
            "error": """docker build -t myapp .
Step 5/10 : RUN npm install
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! Found: react@18.2.0
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0" from @material-ui/core@4.12.4""",
            "category": "build"
        },
        {
            "scenario": "Container cannot connect to host database",
            "error": """docker logs app-container
Error: connect ECONNREFUSED 127.0.0.1:5432
    at TCPConnectWrap.afterConnect [as oncomplete] (net.js:1141:16)
FATAL: Connection to database failed""",
            "category": "networking"
        },
        {
            "scenario": "Docker daemon not running",
            "error": """docker ps
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?""",
            "category": "daemon"
        },
        {
            "scenario": "No space left on device",
            "error": """docker build -t myapp .
Error: write /var/lib/docker/tmp/xxx: no space left on device
ERROR: failed to solve: failed to compute cache key: failed to copy: write /var/lib/docker/tmp/xxx: no space left on device""",
            "category": "storage"
        },
        {
            "scenario": "Port already in use",
            "error": """docker run -p 3000:3000 myapp
docker: Error response from daemon: driver failed programming external connectivity on endpoint myapp: Bind for 0.0.0.0:3000 failed: port is already allocated.""",
            "category": "networking"
        },
        {
            "scenario": "Permission denied on volume mount",
            "error": """docker run -v /data:/app/data myapp
Error: EACCES: permission denied, open '/app/data/config.json'
    at Object.openSync (fs.js:476:3)""",
            "category": "permissions"
        },
        {
            "scenario": "Docker compose service unhealthy",
            "error": """docker-compose up
Creating network "app_default" with the default driver
Creating app_db_1    ... done
Creating app_redis_1 ... done
Creating app_api_1   ... done
ERROR: for app_api_1  Container "xxx" is unhealthy.

docker logs app_api_1
Waiting for database connection...
Connection timeout after 30s""",
            "category": "compose"
        },
    ],
    "terraform": [
        {
            "scenario": "State lock error",
            "error": """terraform apply
Error: Error acquiring the state lock

Error message: ConditionalCheckFailedException: The conditional request failed
Lock Info:
  ID:        abc-123-def
  Path:      s3://my-bucket/terraform.tfstate
  Operation: OperationTypeApply
  Who:       user@host
  Created:   2024-01-15 10:30:00 UTC""",
            "category": "state"
        },
        {
            "scenario": "Resource already exists",
            "error": """terraform apply
Error: error creating S3 bucket (my-bucket-name): BucketAlreadyExists: The requested bucket name is not available. The bucket namespace is shared by all users of the system.""",
            "category": "resource"
        },
        {
            "scenario": "Provider authentication failed",
            "error": """terraform plan
Error: error configuring Terraform AWS Provider: error validating provider credentials: error calling sts:GetCallerIdentity: operation error STS: GetCallerIdentity, https response error StatusCode: 403, api error InvalidClientTokenId: The security token included in the request is invalid.""",
            "category": "auth"
        },
        {
            "scenario": "Dependency cycle",
            "error": """terraform plan
Error: Cycle: aws_security_group.web, aws_security_group.db, aws_security_group.web

A cycle exists between these resources. Terraform cannot determine the order to create them.""",
            "category": "dependency"
        },
        {
            "scenario": "Invalid resource reference",
            "error": """terraform plan
Error: Reference to undeclared resource

  on main.tf line 25, in resource "aws_instance" "web":
  25:   subnet_id = aws_subnet.private.id

A managed resource "aws_subnet" "private" has not been declared in the root module.""",
            "category": "syntax"
        },
        {
            "scenario": "State drift detected",
            "error": """terraform plan
Note: Objects have changed outside of Terraform

Terraform detected the following changes made outside of Terraform since the last "terraform apply":

  # aws_instance.web has been changed
  ~ resource "aws_instance" "web" {
      ~ instance_type = "t3.micro" -> "t3.small"
    }""",
            "category": "drift"
        },
    ],
    "azure": [
        {
            "scenario": "Resource group deployment failed",
            "error": """az deployment group create --resource-group myRG --template-file main.bicep
{"error":{"code":"InvalidTemplateDeployment","message":"The template deployment 'main' is not valid according to the validation procedure. The tracking id is 'xxx'. See inner errors for details.","details":[{"code":"RequestDisallowedByPolicy","message":"Resource 'storage-account' was disallowed by policy."}]}}""",
            "category": "policy"
        },
        {
            "scenario": "Azure CLI authentication expired",
            "error": """az vm list
AADSTS700082: The refresh token has expired due to inactivity. The token was issued on 2024-01-01 and was inactive for 90 days.
Please run 'az login' to setup account.""",
            "category": "auth"
        },
        {
            "scenario": "Quota exceeded",
            "error": """az vm create --name myVM --resource-group myRG --image Ubuntu2204
(OperationNotAllowed) Operation could not be completed as it results in exceeding approved Total Regional Cores quota. Current limit: 10, Current usage: 10, Additional required: 4.""",
            "category": "quota"
        },
        {
            "scenario": "Storage account name taken",
            "error": """az storage account create --name mystorageaccount --resource-group myRG
(StorageAccountAlreadyTaken) The storage account named mystorageaccount is already taken.""",
            "category": "naming"
        },
        {
            "scenario": "Function app deployment failed",
            "error": """func azure functionapp publish myfuncapp
Getting site publishing info...
Uploading 45.2 MB [###########]
Remote build failed.
Error: Could not find a part of the path '/home/site/wwwroot/node_modules'.""",
            "category": "deployment"
        },
    ],
    "gcp": [
        {
            "scenario": "IAM permission denied",
            "error": """gcloud compute instances create my-vm --zone=us-central1-a
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Required 'compute.instances.create' permission for 'projects/my-project/zones/us-central1-a/instances/my-vm'""",
            "category": "iam"
        },
        {
            "scenario": "API not enabled",
            "error": """gcloud run deploy my-service --image gcr.io/my-project/app
ERROR: (gcloud.run.deploy) PERMISSION_DENIED: Cloud Run API has not been used in project my-project before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/run.googleapis.com""",
            "category": "api"
        },
        {
            "scenario": "Quota exceeded",
            "error": """gcloud compute instances create my-vm --machine-type=n1-highmem-64
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Quota 'CPUS' exceeded. Limit: 24.0 in region us-central1.""",
            "category": "quota"
        },
        {
            "scenario": "Cloud Function deployment timeout",
            "error": """gcloud functions deploy my-function --runtime python39 --trigger-http
Deploying function (may take a while - up to 2 minutes)...failed.
ERROR: (gcloud.functions.deploy) OperationError: code=3, message=Build failed: npm ERR! code ETIMEDOUT""",
            "category": "deployment"
        },
    ],
    "nodejs": [
        {
            "scenario": "Module not found",
            "error": """node app.js
Error: Cannot find module 'express'
Require stack:
- /app/server.js
    at Function.Module._resolveFilename (node:internal/modules/cjs/loader:933:15)""",
            "category": "module"
        },
        {
            "scenario": "Port already in use",
            "error": """node server.js
Error: listen EADDRINUSE: address already in use :::3000
    at Server.setupListenHandle [as _listen2] (node:net:1330:16)""",
            "category": "port"
        },
        {
            "scenario": "Heap out of memory",
            "error": """node --max-old-space-size=512 process.js
FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
 1: 0x100a3c0d8 node::Abort()
 2: 0x100a3c254 node::OnFatalError""",
            "category": "memory"
        },
        {
            "scenario": "Unhandled promise rejection",
            "error": """node app.js
(node:1234) UnhandledPromiseRejectionWarning: Error: Connection refused
    at TCPConnectWrap.afterConnect [as oncomplete]
(node:1234) UnhandledPromiseRejectionWarning: Unhandled promise rejection. This error originated either by throwing inside of an async function without a catch block""",
            "category": "async"
        },
        {
            "scenario": "npm install EACCES",
            "error": """npm install -g typescript
npm ERR! code EACCES
npm ERR! syscall mkdir
npm ERR! path /usr/local/lib/node_modules/typescript
npm ERR! errno -13
npm ERR! Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'""",
            "category": "permissions"
        },
    ],
    "redis": [
        {
            "scenario": "Connection refused",
            "error": """redis-cli ping
Could not connect to Redis at 127.0.0.1:6379: Connection refused""",
            "category": "connection"
        },
        {
            "scenario": "Max memory reached",
            "error": """redis-cli SET mykey "value"
(error) OOM command not allowed when used memory > 'maxmemory'.""",
            "category": "memory"
        },
        {
            "scenario": "Authentication required",
            "error": """redis-cli GET mykey
(error) NOAUTH Authentication required.""",
            "category": "auth"
        },
        {
            "scenario": "Cluster node failure",
            "error": """redis-cli cluster info
cluster_state:fail
cluster_slots_assigned:16384
cluster_slots_ok:10923
cluster_slots_pfail:0
cluster_slots_fail:5461
cluster_known_nodes:6
cluster_size:3""",
            "category": "cluster"
        },
        {
            "scenario": "RDB save error",
            "error": """tail /var/log/redis/redis-server.log
[1234] 15 Jan 10:30:00.000 * Background saving started by pid 5678
[1234] 15 Jan 10:30:05.000 # Background saving error
[1234] 15 Jan 10:30:05.000 # MISCONF Redis is configured to save RDB snapshots, but it is currently not able to persist on disk.""",
            "category": "persistence"
        },
    ],
    "mongodb": [
        {
            "scenario": "Authentication failed",
            "error": """mongosh "mongodb://localhost:27017/mydb" -u admin -p
MongoServerError: Authentication failed.
    at Connection.onMessage (/usr/local/lib/node_modules/mongosh/node_modules/mongodb/lib/cmap/connection.js:230:26)""",
            "category": "auth"
        },
        {
            "scenario": "Replica set election",
            "error": """mongosh
MongoServerSelectionError: connection timed out
Topology description: ReplicaSetNoPrimary, servers: [
  { address: 'mongo1:27017', type: 'RSGhost' },
  { address: 'mongo2:27017', type: 'RSGhost' },
  { address: 'mongo3:27017', type: 'RSGhost' }
]""",
            "category": "replication"
        },
        {
            "scenario": "Write concern timeout",
            "error": """db.collection.insertOne({...})
MongoWriteConcernError: waiting for replication timed out
  writeConcernError: {
    code: 64,
    codeName: 'WriteConcernFailed',
    errmsg: 'waiting for replication timed out',
    errInfo: { wtimeout: true }
  }""",
            "category": "write_concern"
        },
        {
            "scenario": "Disk space full",
            "error": """mongosh
MongoServerError: Disk space is critically low. Only 1% free space remaining.
Error code: 14031
Cannot write to database.""",
            "category": "storage"
        },
    ],
    "nginx": [
        {
            "scenario": "Configuration test failed",
            "error": """nginx -t
nginx: [emerg] unknown directive "proxy_passs" in /etc/nginx/conf.d/default.conf:15
nginx: configuration file /etc/nginx/nginx.conf test failed""",
            "category": "config"
        },
        {
            "scenario": "502 Bad Gateway upstream",
            "error": """tail /var/log/nginx/error.log
2024/01/15 10:30:00 [error] 1234#1234: *1 connect() failed (111: Connection refused) while connecting to upstream, client: 10.0.0.1, server: example.com, request: "GET / HTTP/1.1", upstream: "http://127.0.0.1:3000/", host: "example.com" """,
            "category": "upstream"
        },
        {
            "scenario": "SSL certificate expired",
            "error": """curl https://example.com
curl: (60) SSL certificate problem: certificate has expired
More details here: https://curl.haxx.se/docs/sslcerts.html

nginx error.log:
SSL_do_handshake() failed (SSL: error:0A000086:SSL routines::certificate verify failed:certificate has expired)""",
            "category": "ssl"
        },
        {
            "scenario": "Permission denied on socket",
            "error": """nginx -t
nginx: [crit] *1 connect() to unix:/var/run/php/php8.1-fpm.sock failed (13: Permission denied) while connecting to upstream""",
            "category": "permissions"
        },
        {
            "scenario": "Too many open files",
            "error": """tail /var/log/nginx/error.log
2024/01/15 10:30:00 [alert] 1234#1234: *100000 socket() failed (24: Too many open files) while connecting to upstream""",
            "category": "limits"
        },
    ],
    "postgresql": [
        {
            "scenario": "Too many connections",
            "error": """psql -U postgres
FATAL: too many connections for role "postgres"
DETAIL: max_connections is 100 but 100 connections are already open.""",
            "category": "connections"
        },
        {
            "scenario": "Permission denied on table",
            "error": """SELECT * FROM users;
ERROR:  permission denied for table users
HINT:  Grant SELECT permission on the table to the role.""",
            "category": "permissions"
        },
        {
            "scenario": "Deadlock detected",
            "error": """ERROR:  deadlock detected
DETAIL:  Process 1234 waits for ShareLock on transaction 5678; blocked by process 9012.
Process 9012 waits for ShareLock on transaction 1234; blocked by process 1234.
HINT:  See server log for query details.
CONTEXT:  while updating tuple (0,1) in relation "accounts" """,
            "category": "deadlock"
        },
        {
            "scenario": "Disk full",
            "error": """INSERT INTO logs VALUES (...);
ERROR:  could not extend file "base/16384/16385": No space left on device
HINT:  Check free disk space.""",
            "category": "storage"
        },
        {
            "scenario": "Replication lag",
            "error": """SELECT client_addr, state, sent_lsn, write_lsn, replay_lag 
FROM pg_stat_replication;
 client_addr | state   | sent_lsn  | write_lsn | replay_lag
-------------+---------+-----------+-----------+------------
 10.0.0.2    | catchup | 0/5000000 | 0/3000000 | 02:30:00""",
            "category": "replication"
        },
    ],
    "influxdb": [
        {
            "scenario": "Write timeout",
            "error": """influx write -b mybucket -f data.csv
Error: failed to write data: timeout: context deadline exceeded
Partial write occurred. 50000 of 100000 points written.""",
            "category": "write"
        },
        {
            "scenario": "Query killed - memory",
            "error": """influx query 'from(bucket:"metrics") |> range(start: -30d) |> filter(fn: (r) => r._measurement == "cpu")'
Error: query terminated: memory allocation limit reached: 1073741824 bytes""",
            "category": "query"
        },
        {
            "scenario": "Token invalid",
            "error": """influx bucket list
Error: failed to list buckets: unauthorized: unauthorized access
Hint: Check that your token has the correct permissions and has not expired.""",
            "category": "auth"
        },
        {
            "scenario": "Retention policy issue",
            "error": """influx bucket update -n mybucket --retention 7d
Error: failed to update bucket: retention policy duration too short: minimum 1h
Current shard group duration: 1d requires retention >= 1d""",
            "category": "retention"
        },
    ],
}

# =============================================================================
# RESPONSE FORMAT
# =============================================================================

RESPONSE_FORMAT = """**Root Cause:** [Specific diagnosis of what's causing this issue]

**Severity:** [Low/Medium/High/Critical]

**Immediate Fix:**
1. [First step]
2. [Second step]
3. [Additional steps as needed]

**Prevention:** [How to prevent this issue in the future]"""

# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_incident_response(client: anthropic.Anthropic, tech: str, scenario: str, error: str, category: str) -> str:
    """Generate an expert incident response using Claude Haiku."""
    
    prompt = f"""You are a senior DevOps/SRE engineer. Analyze this {tech} incident and provide a diagnosis and fix.

**Scenario:** {scenario}

**Error/Logs:**
```
{error}
```

Respond in this exact format:

**Root Cause:** [1-2 sentences explaining the specific cause]

**Severity:** [One of: Low, Medium, High, Critical]

**Immediate Fix:**
1. [Specific command or action]
2. [Next step]
3. [Additional steps if needed]

**Prevention:** [1-2 sentences on how to prevent this]

Requirements:
- Be specific and actionable
- Include exact commands where applicable
- Keep it concise but complete
- Don't use generic advice"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()


def generate_synthetic_dataset(target_count: int = TARGET_EXAMPLES):
    """Generate synthetic incident response training data."""
    
    client = anthropic.Anthropic()
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Calculate examples per tech
    tech_count = len(INCIDENT_TEMPLATES)
    examples_per_tech = target_count // tech_count
    
    print("=" * 60)
    print("SYNTHETIC INCIDENT DATA GENERATOR")
    print("=" * 60)
    print(f"Target examples: {target_count}")
    print(f"Technologies: {tech_count}")
    print(f"Examples per tech: ~{examples_per_tech}")
    print(f"Model: {MODEL}")
    print()
    
    examples = []
    stats = {"total": 0, "by_tech": {}}
    
    for tech, templates in INCIDENT_TEMPLATES.items():
        print(f"\n[{tech.upper()}]")
        stats["by_tech"][tech] = 0
        
        # Calculate how many to generate for this tech
        tech_target = examples_per_tech
        examples_per_template = max(tech_target // len(templates), 1)
        
        for template in templates:
            for i in range(examples_per_template):
                if stats["total"] >= target_count:
                    break
                
                try:
                    response = generate_incident_response(
                        client,
                        tech,
                        template["scenario"],
                        template["error"],
                        template["category"]
                    )
                    
                    # Format as training example
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert DevOps engineer and SRE. Analyze the provided error logs, stack traces, or incident descriptions.\n\nYour response should include:\n1. **Root Cause**: What is causing this issue\n2. **Severity**: Low / Medium / High / Critical\n3. **Fix**: Step-by-step solution to resolve the issue\n4. **Prevention**: How to prevent this in the future (optional)\n\nBe direct, specific, and actionable. Reference exact commands, config changes, or code fixes when applicable."
                            },
                            {
                                "role": "user",
                                "content": f"Analyze this {tech} incident and provide diagnosis and fix:\n\n```\n{template['error']}\n```"
                            },
                            {
                                "role": "assistant",
                                "content": response
                            }
                        ],
                        "_meta": {
                            "source": "synthetic",
                            "tech": tech,
                            "scenario": template["scenario"],
                            "category": template["category"]
                        }
                    }
                    
                    examples.append(example)
                    stats["total"] += 1
                    stats["by_tech"][tech] += 1
                    
                    if stats["total"] % 50 == 0:
                        print(f"  Generated {stats['total']}/{target_count}...")
                        save_checkpoint(examples, stats)
                    
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    time.sleep(2)
                    continue
            
            if stats["total"] >= target_count:
                break
        
        print(f"  {tech}: {stats['by_tech'][tech]} examples")
        
        if stats["total"] >= target_count:
            break
    
    # Final save
    save_checkpoint(examples, stats, final=True)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {stats['total']}")
    print(f"\nBy technology:")
    for tech, count in stats["by_tech"].items():
        print(f"  {tech}: {count}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    return examples, stats


def save_checkpoint(examples: list, stats: dict, final: bool = False):
    """Save checkpoint of generated examples."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": MODEL,
        "stats": stats,
        "examples": examples
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)


def merge_with_training_data(
    synthetic_file: str = OUTPUT_FILE,
    train_file: str = "data/processed/train.jsonl",
    output_file: str = "data/processed/train_with_synthetic.jsonl"
):
    """Merge synthetic data with existing training data."""
    
    print("\nMerging synthetic data with training data...")
    
    # Load synthetic
    with open(synthetic_file) as f:
        synthetic_data = json.load(f)
    synthetic_examples = synthetic_data["examples"]
    print(f"Synthetic examples: {len(synthetic_examples)}")
    
    # Load existing training data
    existing = []
    with open(train_file) as f:
        for line in f:
            existing.append(json.loads(line))
    print(f"Existing examples: {len(existing)}")
    
    # Combine and shuffle
    combined = existing + synthetic_examples
    import random
    random.seed(42)
    random.shuffle(combined)
    
    # Save
    with open(output_file, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Combined total: {len(combined)}")
    print(f"Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic incident data")
    parser.add_argument("--count", type=int, default=TARGET_EXAMPLES, help="Number of examples")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing synthetic data")
    
    args = parser.parse_args()
    
    if args.merge_only:
        merge_with_training_data()
    else:
        generate_synthetic_dataset(args.count)
        print("\nTo merge with training data, run:")
        print("  python generate_synthetic.py --merge-only")
