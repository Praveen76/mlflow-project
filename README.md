


## 1) Log in to ECR (use command substitution to avoid the TTY error)

Use **either** of these. The second one avoids the `stdin` pipe:

```bash
# A) standard (works in most shells)
aws ecr get-login-password --region us-east-1 \
| docker login --username AWS --password-stdin 481181974576.dkr.ecr.us-east-1.amazonaws.com

# B) fallback (avoids stdin/TTY issues)
docker login \
  --username AWS \
  --password "$(aws ecr get-login-password --region us-east-1)" \
  481181974576.dkr.ecr.us-east-1.amazonaws.com
```

You should see: `Login Succeeded`.

---

## 2) Build, tag, push your MLflow image

```bash
# build from your MLflow Dockerfile
docker build -f Dockerfile.mlflow -t mlflow-ecr .

# tag with latest and commit sha
SHA=$(git rev-parse --short HEAD || echo local)
docker tag mlflow-ecr:latest 481181974576.dkr.ecr.us-east-1.amazonaws.com/mlflow-ecr:latest
docker tag mlflow-ecr:latest 481181974576.dkr.ecr.us-east-1.amazonaws.com/mlflow-ecr:$SHA

# push
docker push 481181974576.dkr.ecr.us-east-1.amazonaws.com/mlflow-ecr:latest
docker push 481181974576.dkr.ecr.us-east-1.amazonaws.com/mlflow-ecr:$SHA
```

