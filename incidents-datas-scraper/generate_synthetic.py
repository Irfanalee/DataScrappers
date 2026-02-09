"""
Synthetic Data Generator for DevOps Incident Responder
Optimized with batch generation to reduce API costs

Uses batched generation: 5 responses per API call
Estimated cost: ~$0.80 for 1,500 examples (vs $2.50 without batching)
"""

import anthropic
import json
import random
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "data/synthetic"
OUTPUT_FILE = f"{OUTPUT_DIR}/synthetic_incidents.json"
TARGET_EXAMPLES = 1500
MODEL = "claude-3-5-haiku-20241022"

# Batch size - generate multiple responses per API call
BATCH_SIZE = 5

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
            "category": "pod_failure",
            "hints": ["missing entrypoint", "incorrect workdir", "missing dependencies", "wrong CMD", "volume mount issue"]
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
            "category": "resource",
            "hints": ["memory limit too low", "memory leak", "large dataset processing", "unbounded cache", "JVM heap misconfigured"]
        },
        {
            "scenario": "ImagePullBackOff",
            "error": """kubectl get pods
NAME                    READY   STATUS             RESTARTS   AGE
web-app-6c7d8e9f0-xyz   0/1     ImagePullBackOff   0          5m

kubectl describe pod web-app-6c7d8e9f0-xyz
Events:
  Warning  Failed   Failed to pull image "myregistry.io/app:v2.1": rpc error: code = Unknown desc = Error response from daemon: unauthorized: authentication required""",
            "category": "image",
            "hints": ["missing imagePullSecrets", "expired credentials", "wrong registry URL", "image tag doesn't exist", "private registry auth"]
        },
        {
            "scenario": "Service not reachable",
            "error": """kubectl exec -it debug-pod -- curl http://backend-service:8080
curl: (7) Failed to connect to backend-service port 8080: Connection refused

kubectl get endpoints backend-service
NAME              ENDPOINTS   AGE
backend-service   <none>      15m""",
            "category": "networking",
            "hints": ["selector mismatch", "pods not ready", "wrong port", "network policy blocking", "service in wrong namespace"]
        },
        {
            "scenario": "PersistentVolumeClaim pending",
            "error": """kubectl get pvc
NAME        STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   AGE
data-pvc    Pending                                      standard       10m

kubectl describe pvc data-pvc
Events:
  Warning  ProvisioningFailed  storageclass.storage.k8s.io "standard" not found""",
            "category": "storage",
            "hints": ["storageclass missing", "no available PV", "access mode mismatch", "capacity too large", "zone constraints"]
        },
        {
            "scenario": "RBAC permission denied",
            "error": """kubectl get pods -n production
Error from server (Forbidden): pods is forbidden: User "developer@company.com" cannot list resource "pods" in API group "" in the namespace "production" """,
            "category": "rbac",
            "hints": ["missing RoleBinding", "wrong namespace", "ClusterRole needed", "service account permissions", "RBAC not configured"]
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
            "category": "node",
            "hints": ["disk full", "memory exhausted", "kubelet crashed", "network unreachable", "docker daemon issues"]
        },
        {
            "scenario": "Ingress 502 Bad Gateway",
            "error": """curl -I https://app.example.com
HTTP/2 502
server: nginx

kubectl logs -n ingress-nginx ingress-nginx-controller-xxx
upstream connect error or disconnect/reset before headers. reset reason: connection termination""",
            "category": "ingress",
            "hints": ["backend service down", "health check failing", "timeout too short", "SSL termination issue", "wrong backend port"]
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
            "category": "build",
            "hints": ["peer dependency conflict", "npm version issue", "lock file outdated", "use --legacy-peer-deps", "upgrade packages"]
        },
        {
            "scenario": "Container cannot connect to host database",
            "error": """docker logs app-container
Error: connect ECONNREFUSED 127.0.0.1:5432
    at TCPConnectWrap.afterConnect [as oncomplete] (net.js:1141:16)
FATAL: Connection to database failed""",
            "category": "networking",
            "hints": ["use host.docker.internal", "wrong network mode", "database not exposed", "firewall blocking", "use docker network"]
        },
        {
            "scenario": "Docker daemon not running",
            "error": """docker ps
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?""",
            "category": "daemon",
            "hints": ["start docker service", "socket permissions", "systemd service failed", "docker desktop not running", "WSL integration"]
        },
        {
            "scenario": "No space left on device",
            "error": """docker build -t myapp .
Error: write /var/lib/docker/tmp/xxx: no space left on device
ERROR: failed to solve: failed to compute cache key: failed to copy: write /var/lib/docker/tmp/xxx: no space left on device""",
            "category": "storage",
            "hints": ["docker system prune", "remove unused images", "increase disk", "clear build cache", "check overlay2 usage"]
        },
        {
            "scenario": "Port already in use",
            "error": """docker run -p 3000:3000 myapp
docker: Error response from daemon: driver failed programming external connectivity on endpoint myapp: Bind for 0.0.0.0:3000 failed: port is already allocated.""",
            "category": "networking",
            "hints": ["find process using port", "stop conflicting container", "use different port", "check docker-compose", "lsof -i :3000"]
        },
        {
            "scenario": "Permission denied on volume mount",
            "error": """docker run -v /data:/app/data myapp
Error: EACCES: permission denied, open '/app/data/config.json'
    at Object.openSync (fs.js:476:3)""",
            "category": "permissions",
            "hints": ["fix host permissions", "run as correct user", "use :z or :Z flag", "chown in dockerfile", "USER directive mismatch"]
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
            "category": "compose",
            "hints": ["add depends_on with condition", "healthcheck misconfigured", "increase timeout", "database not ready", "use wait-for script"]
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
            "category": "state",
            "hints": ["force-unlock with ID", "previous run crashed", "concurrent execution", "CI/CD conflict", "check DynamoDB table"]
        },
        {
            "scenario": "Resource already exists",
            "error": """terraform apply
Error: error creating S3 bucket (my-bucket-name): BucketAlreadyExists: The requested bucket name is not available. The bucket namespace is shared by all users of the system.""",
            "category": "resource",
            "hints": ["import existing resource", "use unique naming", "check if managed elsewhere", "state mismatch", "use random suffix"]
        },
        {
            "scenario": "Provider authentication failed",
            "error": """terraform plan
Error: error configuring Terraform AWS Provider: error validating provider credentials: error calling sts:GetCallerIdentity: operation error STS: GetCallerIdentity, https response error StatusCode: 403, api error InvalidClientTokenId: The security token included in the request is invalid.""",
            "category": "auth",
            "hints": ["expired credentials", "wrong profile", "missing env vars", "IAM permissions", "MFA required"]
        },
        {
            "scenario": "Dependency cycle",
            "error": """terraform plan
Error: Cycle: aws_security_group.web, aws_security_group.db, aws_security_group.web

A cycle exists between these resources. Terraform cannot determine the order to create them.""",
            "category": "dependency",
            "hints": ["use security_group_rule", "separate ingress/egress", "break circular reference", "use depends_on carefully", "refactor module"]
        },
        {
            "scenario": "Invalid resource reference",
            "error": """terraform plan
Error: Reference to undeclared resource

  on main.tf line 25, in resource "aws_instance" "web":
  25:   subnet_id = aws_subnet.private.id

A managed resource "aws_subnet" "private" has not been declared in the root module.""",
            "category": "syntax",
            "hints": ["typo in resource name", "missing module output", "resource not created", "wrong module reference", "check variable names"]
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
            "category": "drift",
            "hints": ["manual console changes", "another tool modified", "refresh state", "import changes", "enforce IaC policy"]
        },
    ],
    "azure": [
        {
            "scenario": "Resource group deployment failed",
            "error": """az deployment group create --resource-group myRG --template-file main.bicep
{"error":{"code":"InvalidTemplateDeployment","message":"The template deployment 'main' is not valid according to the validation procedure. The tracking id is 'xxx'. See inner errors for details.","details":[{"code":"RequestDisallowedByPolicy","message":"Resource 'storage-account' was disallowed by policy."}]}}""",
            "category": "policy",
            "hints": ["Azure Policy blocking", "naming convention violation", "region restriction", "SKU not allowed", "check policy assignments"]
        },
        {
            "scenario": "Azure CLI authentication expired",
            "error": """az vm list
AADSTS700082: The refresh token has expired due to inactivity. The token was issued on 2024-01-01 and was inactive for 90 days.
Please run 'az login' to setup account.""",
            "category": "auth",
            "hints": ["az login", "service principal expired", "token cache cleared", "MFA required", "use managed identity"]
        },
        {
            "scenario": "Quota exceeded",
            "error": """az vm create --name myVM --resource-group myRG --image Ubuntu2204
(OperationNotAllowed) Operation could not be completed as it results in exceeding approved Total Regional Cores quota. Current limit: 10, Current usage: 10, Additional required: 4.""",
            "category": "quota",
            "hints": ["request quota increase", "use different region", "use smaller VM", "clean up unused resources", "check subscription limits"]
        },
        {
            "scenario": "Storage account name taken",
            "error": """az storage account create --name mystorageaccount --resource-group myRG
(StorageAccountAlreadyTaken) The storage account named mystorageaccount is already taken.""",
            "category": "naming",
            "hints": ["globally unique name required", "add random suffix", "use naming convention", "check existing accounts", "different name"]
        },
        {
            "scenario": "Function app deployment failed",
            "error": """func azure functionapp publish myfuncapp
Getting site publishing info...
Uploading 45.2 MB [###########]
Remote build failed.
Error: Could not find a part of the path '/home/site/wwwroot/node_modules'.""",
            "category": "deployment",
            "hints": ["remote build issue", "package.json missing", "run npm install locally", "use zip deploy", "check runtime version"]
        },
    ],
    "gcp": [
        {
            "scenario": "IAM permission denied",
            "error": """gcloud compute instances create my-vm --zone=us-central1-a
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Required 'compute.instances.create' permission for 'projects/my-project/zones/us-central1-a/instances/my-vm'""",
            "category": "iam",
            "hints": ["missing IAM role", "wrong project", "service account permissions", "org policy restriction", "check IAM bindings"]
        },
        {
            "scenario": "API not enabled",
            "error": """gcloud run deploy my-service --image gcr.io/my-project/app
ERROR: (gcloud.run.deploy) PERMISSION_DENIED: Cloud Run API has not been used in project my-project before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/run.googleapis.com""",
            "category": "api",
            "hints": ["enable API", "gcloud services enable", "billing not enabled", "project selector wrong", "API quota exceeded"]
        },
        {
            "scenario": "Quota exceeded",
            "error": """gcloud compute instances create my-vm --machine-type=n1-highmem-64
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Quota 'CPUS' exceeded. Limit: 24.0 in region us-central1.""",
            "category": "quota",
            "hints": ["request quota increase", "use different zone", "smaller machine type", "preemptible instances", "check quotas page"]
        },
        {
            "scenario": "Cloud Function deployment timeout",
            "error": """gcloud functions deploy my-function --runtime python39 --trigger-http
Deploying function (may take a while - up to 2 minutes)...failed.
ERROR: (gcloud.functions.deploy) OperationError: code=3, message=Build failed: npm ERR! code ETIMEDOUT""",
            "category": "deployment",
            "hints": ["network timeout", "increase memory", "check requirements.txt", "private dependencies", "use artifact registry"]
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
            "category": "module",
            "hints": ["npm install", "missing package.json", "node_modules deleted", "wrong working directory", "check import path"]
        },
        {
            "scenario": "Port already in use",
            "error": """node server.js
Error: listen EADDRINUSE: address already in use :::3000
    at Server.setupListenHandle [as _listen2] (node:net:1330:16)""",
            "category": "port",
            "hints": ["kill existing process", "use different port", "lsof -i :3000", "check for zombie process", "PORT env variable"]
        },
        {
            "scenario": "Heap out of memory",
            "error": """node --max-old-space-size=512 process.js
FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
 1: 0x100a3c0d8 node::Abort()
 2: 0x100a3c254 node::OnFatalError""",
            "category": "memory",
            "hints": ["increase --max-old-space-size", "memory leak", "stream large files", "pagination", "garbage collection"]
        },
        {
            "scenario": "Unhandled promise rejection",
            "error": """node app.js
(node:1234) UnhandledPromiseRejectionWarning: Error: Connection refused
    at TCPConnectWrap.afterConnect [as oncomplete]
(node:1234) UnhandledPromiseRejectionWarning: Unhandled promise rejection. This error originated either by throwing inside of an async function without a catch block""",
            "category": "async",
            "hints": ["add try/catch", "use .catch()", "global error handler", "async/await wrapper", "process.on unhandledRejection"]
        },
        {
            "scenario": "npm install EACCES",
            "error": """npm install -g typescript
npm ERR! code EACCES
npm ERR! syscall mkdir
npm ERR! path /usr/local/lib/node_modules/typescript
npm ERR! errno -13
npm ERR! Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'""",
            "category": "permissions",
            "hints": ["use nvm", "fix npm permissions", "avoid sudo", "use npx instead", "change npm prefix"]
        },
    ],
    "redis": [
        {
            "scenario": "Connection refused",
            "error": """redis-cli ping
Could not connect to Redis at 127.0.0.1:6379: Connection refused""",
            "category": "connection",
            "hints": ["redis not running", "wrong host/port", "firewall blocking", "bind address issue", "protected mode"]
        },
        {
            "scenario": "Max memory reached",
            "error": """redis-cli SET mykey "value"
(error) OOM command not allowed when used memory > 'maxmemory'.""",
            "category": "memory",
            "hints": ["increase maxmemory", "set eviction policy", "clear unused keys", "memory fragmentation", "check memory usage"]
        },
        {
            "scenario": "Authentication required",
            "error": """redis-cli GET mykey
(error) NOAUTH Authentication required.""",
            "category": "auth",
            "hints": ["AUTH command", "redis.conf requirepass", "connection string password", "ACL user", "check credentials"]
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
            "category": "cluster",
            "hints": ["node down", "fix or forget node", "cluster rebalance", "check replica promotion", "quorum lost"]
        },
        {
            "scenario": "RDB save error",
            "error": """tail /var/log/redis/redis-server.log
[1234] 15 Jan 10:30:00.000 * Background saving started by pid 5678
[1234] 15 Jan 10:30:05.000 # Background saving error
[1234] 15 Jan 10:30:05.000 # MISCONF Redis is configured to save RDB snapshots, but it is currently not able to persist on disk.""",
            "category": "persistence",
            "hints": ["disk full", "permissions issue", "disable persistence", "check rdb directory", "stop-writes-on-bgsave-error"]
        },
    ],
    "mongodb": [
        {
            "scenario": "Authentication failed",
            "error": """mongosh "mongodb://localhost:27017/mydb" -u admin -p
MongoServerError: Authentication failed.
    at Connection.onMessage (/usr/local/lib/node_modules/mongosh/node_modules/mongodb/lib/cmap/connection.js:230:26)""",
            "category": "auth",
            "hints": ["wrong credentials", "user not in database", "authSource parameter", "SCRAM mechanism", "check admin database"]
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
            "category": "replication",
            "hints": ["no primary elected", "network partition", "majority unavailable", "check rs.status()", "force reconfigure"]
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
            "category": "write_concern",
            "hints": ["replica lag", "secondary down", "reduce write concern", "increase wtimeout", "check replication status"]
        },
        {
            "scenario": "Disk space full",
            "error": """mongosh
MongoServerError: Disk space is critically low. Only 1% free space remaining.
Error code: 14031
Cannot write to database.""",
            "category": "storage",
            "hints": ["add disk space", "compact collections", "drop old data", "enable compression", "archive to cold storage"]
        },
    ],
    "nginx": [
        {
            "scenario": "Configuration test failed",
            "error": """nginx -t
nginx: [emerg] unknown directive "proxy_passs" in /etc/nginx/conf.d/default.conf:15
nginx: configuration file /etc/nginx/nginx.conf test failed""",
            "category": "config",
            "hints": ["typo in directive", "syntax error", "missing semicolon", "wrong block", "check nginx docs"]
        },
        {
            "scenario": "502 Bad Gateway upstream",
            "error": """tail /var/log/nginx/error.log
2024/01/15 10:30:00 [error] 1234#1234: *1 connect() failed (111: Connection refused) while connecting to upstream, client: 10.0.0.1, server: example.com, request: "GET / HTTP/1.1", upstream: "http://127.0.0.1:3000/", host: "example.com" """,
            "category": "upstream",
            "hints": ["backend not running", "wrong upstream port", "socket vs http", "health check failing", "firewall blocking"]
        },
        {
            "scenario": "SSL certificate expired",
            "error": """curl https://example.com
curl: (60) SSL certificate problem: certificate has expired
More details here: https://curl.haxx.se/docs/sslcerts.html

nginx error.log:
SSL_do_handshake() failed (SSL: error:0A000086:SSL routines::certificate verify failed:certificate has expired)""",
            "category": "ssl",
            "hints": ["renew certificate", "certbot renew", "check cert dates", "reload nginx", "automate renewal"]
        },
        {
            "scenario": "Permission denied on socket",
            "error": """nginx -t
nginx: [crit] *1 connect() to unix:/var/run/php/php8.1-fpm.sock failed (13: Permission denied) while connecting to upstream""",
            "category": "permissions",
            "hints": ["socket permissions", "nginx user group", "php-fpm pool config", "listen.owner", "selinux context"]
        },
        {
            "scenario": "Too many open files",
            "error": """tail /var/log/nginx/error.log
2024/01/15 10:30:00 [alert] 1234#1234: *100000 socket() failed (24: Too many open files) while connecting to upstream""",
            "category": "limits",
            "hints": ["increase ulimit", "worker_rlimit_nofile", "worker_connections", "system limits", "check /etc/security/limits.conf"]
        },
    ],
    "postgresql": [
        {
            "scenario": "Too many connections",
            "error": """psql -U postgres
FATAL: too many connections for role "postgres"
DETAIL: max_connections is 100 but 100 connections are already open.""",
            "category": "connections",
            "hints": ["increase max_connections", "connection pooler", "pgbouncer", "close idle connections", "check pg_stat_activity"]
        },
        {
            "scenario": "Permission denied on table",
            "error": """SELECT * FROM users;
ERROR:  permission denied for table users
HINT:  Grant SELECT permission on the table to the role.""",
            "category": "permissions",
            "hints": ["GRANT SELECT", "wrong role", "schema permissions", "default privileges", "check pg_roles"]
        },
        {
            "scenario": "Deadlock detected",
            "error": """ERROR:  deadlock detected
DETAIL:  Process 1234 waits for ShareLock on transaction 5678; blocked by process 9012.
Process 9012 waits for ShareLock on transaction 1234; blocked by process 1234.
HINT:  See server log for query details.
CONTEXT:  while updating tuple (0,1) in relation "accounts" """,
            "category": "deadlock",
            "hints": ["transaction ordering", "reduce transaction scope", "row-level locking", "advisory locks", "retry logic"]
        },
        {
            "scenario": "Disk full",
            "error": """INSERT INTO logs VALUES (...);
ERROR:  could not extend file "base/16384/16385": No space left on device
HINT:  Check free disk space.""",
            "category": "storage",
            "hints": ["add disk space", "VACUUM FULL", "delete old data", "pg_repack", "check WAL retention"]
        },
        {
            "scenario": "Replication lag",
            "error": """SELECT client_addr, state, sent_lsn, write_lsn, replay_lag 
FROM pg_stat_replication;
 client_addr | state   | sent_lsn  | write_lsn | replay_lag
-------------+---------+-----------+-----------+------------
 10.0.0.2    | catchup | 0/5000000 | 0/3000000 | 02:30:00""",
            "category": "replication",
            "hints": ["network bandwidth", "replica resources", "long-running queries", "wal_sender_timeout", "check pg_stat_wal_receiver"]
        },
    ],
    "influxdb": [
        {
            "scenario": "Write timeout",
            "error": """influx write -b mybucket -f data.csv
Error: failed to write data: timeout: context deadline exceeded
Partial write occurred. 50000 of 100000 points written.""",
            "category": "write",
            "hints": ["batch size too large", "increase timeout", "check disk I/O", "reduce precision", "async writes"]
        },
        {
            "scenario": "Query killed - memory",
            "error": """influx query 'from(bucket:"metrics") |> range(start: -30d) |> filter(fn: (r) => r._measurement == "cpu")'
Error: query terminated: memory allocation limit reached: 1073741824 bytes""",
            "category": "query",
            "hints": ["narrow time range", "add filters", "use aggregateWindow", "increase memory limit", "downsample data"]
        },
        {
            "scenario": "Token invalid",
            "error": """influx bucket list
Error: failed to list buckets: unauthorized: unauthorized access
Hint: Check that your token has the correct permissions and has not expired.""",
            "category": "auth",
            "hints": ["regenerate token", "check token permissions", "wrong org", "token revoked", "INFLUX_TOKEN env var"]
        },
        {
            "scenario": "Retention policy issue",
            "error": """influx bucket update -n mybucket --retention 7d
Error: failed to update bucket: retention policy duration too short: minimum 1h
Current shard group duration: 1d requires retention >= 1d""",
            "category": "retention",
            "hints": ["retention vs shard duration", "can't reduce below shard", "create new bucket", "data migration", "check current settings"]
        },
    ],
}

# =============================================================================
# SYSTEM PROMPT FOR CONSISTENT FORMAT
# =============================================================================

SYSTEM_PROMPT = """You are an expert DevOps engineer and SRE. Analyze the provided error logs, stack traces, or incident descriptions.

Your response should include:
1. **Root Cause**: What is causing this issue
2. **Severity**: Low / Medium / High / Critical
3. **Fix**: Step-by-step solution to resolve the issue
4. **Prevention**: How to prevent this in the future (optional)

Be direct, specific, and actionable. Reference exact commands, config changes, or code fixes when applicable."""

# =============================================================================
# BATCH GENERATION
# =============================================================================

def generate_batch_responses(
    client: anthropic.Anthropic, 
    tech: str, 
    scenario: str, 
    error: str, 
    hints: list,
    batch_size: int = BATCH_SIZE
) -> list:
    """Generate multiple incident responses in a single API call."""
    
    prompt = f"""You are a senior DevOps/SRE engineer. Generate {batch_size} DIFFERENT expert responses for this {tech} incident.

**Scenario:** {scenario}

**Error/Logs:**
```
{error}
```

**Possible causes to consider (use different ones for variety):** {', '.join(hints)}

Generate {batch_size} different responses. Each response should:
- Focus on a DIFFERENT root cause from the hints
- Have different specific commands/fixes
- Follow this EXACT format

Return ONLY a JSON array with {batch_size} objects, each containing a "response" field.

Example format:
```json
[
  {{"response": "**Root Cause:** [cause 1]\\n\\n**Severity:** High\\n\\n**Immediate Fix:**\\n1. [step]\\n2. [step]\\n\\n**Prevention:** [tip]"}},
  {{"response": "**Root Cause:** [cause 2]\\n\\n**Severity:** Medium\\n\\n**Immediate Fix:**\\n1. [step]\\n2. [step]\\n\\n**Prevention:** [tip]"}}
]
```

Requirements:
- Each response must be UNIQUE with different root cause
- Include exact commands where applicable
- Keep each response concise but complete (150-300 words)
- Return ONLY valid JSON array, no other text"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            json_str = json_match.group()
            responses = json.loads(json_str)
            return [r.get("response", "") for r in responses if r.get("response")]
        else:
            print(f"    Warning: Could not parse JSON from response")
            return []
            
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"    API error: {e}")
        return []


def generate_synthetic_dataset(target_count: int = TARGET_EXAMPLES):
    """Generate synthetic incident response training data using batched generation."""
    
    client = anthropic.Anthropic()
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Count total templates
    total_templates = sum(len(templates) for templates in INCIDENT_TEMPLATES.values())
    batches_per_template = max((target_count // total_templates) // BATCH_SIZE, 1)
    
    print("=" * 60)
    print("SYNTHETIC INCIDENT DATA GENERATOR (BATCHED)")
    print("=" * 60)
    print(f"Target examples: {target_count}")
    print(f"Technologies: {len(INCIDENT_TEMPLATES)}")
    print(f"Total templates: {total_templates}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per template: {batches_per_template}")
    print(f"Estimated API calls: ~{total_templates * batches_per_template}")
    print(f"Model: {MODEL}")
    print()
    
    examples = []
    stats = {"total": 0, "by_tech": {}}
    
    for tech, templates in INCIDENT_TEMPLATES.items():
        print(f"\n[{tech.upper()}]")
        stats["by_tech"][tech] = 0
        
        for template in templates:
            if stats["total"] >= target_count:
                break
            
            for batch_num in range(batches_per_template):
                if stats["total"] >= target_count:
                    break
                
                responses = generate_batch_responses(
                    client,
                    tech,
                    template["scenario"],
                    template["error"],
                    template.get("hints", []),
                    BATCH_SIZE
                )
                
                for response in responses:
                    if stats["total"] >= target_count:
                        break
                    
                    if not response or len(response) < 100:
                        continue
                    
                    # Format as training example
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
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
                
                time.sleep(DELAY_BETWEEN_REQUESTS)
            
            if stats["total"] % 50 == 0 and stats["total"] > 0:
                print(f"  Generated {stats['total']}/{target_count}...")
                save_checkpoint(examples, stats)
        
        print(f"  {tech}: {stats['by_tech'][tech]} examples")
    
    # Final save
    save_checkpoint(examples, stats, final=True)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {stats['total']}")
    print(f"\nBy technology:")
    for tech, count in sorted(stats["by_tech"].items(), key=lambda x: -x[1]):
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
    random.seed(42)
    random.shuffle(combined)
    
    # Save
    with open(output_file, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Combined total: {len(combined)}")
    print(f"Saved to: {output_file}")
    
    # Tech distribution
    tech_counts = {}
    for ex in combined:
        tech = ex.get("_meta", {}).get("tech", "unknown")
        tech_counts[tech] = tech_counts.get(tech, 0) + 1
    
    print(f"\nFinal distribution by tech:")
    for tech, count in sorted(tech_counts.items(), key=lambda x: -x[1]):
        print(f"  {tech}: {count}")
    
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
