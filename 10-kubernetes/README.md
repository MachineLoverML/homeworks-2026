# Homework 10: Kubernetes

Deploys the lead-scoring API from Homework 5 to a local `kind` cluster.

## Questions

**Q1. Local container probability**
Run the HW5 container and `q6_test.py`. What's `conversion_probability` (rounded to 3 decimals)?

**Q2. Environment check**
Run `kind --version` and `kubectl version --client`. Record the output. (Not graded.)

**Q3. Smallest Kubernetes unit**
What is the smallest deployable computing unit Kubernetes manages?
- Node / Pod / Deployment / Service

**Q4. Default service type**
Run `kubectl get services`. What is the `TYPE` of the `kubernetes` service?
- `NodePort` / `ClusterIP` / `ExternalName` / `LoadBalancer`

**Q5. Load image into kind**
Which command loads a local Docker image into a kind cluster?
- `kind create cluster` / `kind build node-image` / `kind load docker-image` / `kubectl apply`

**Q6. Container port**
What container port does `deployment.yaml` use?
- `80` / `8080` / `9000` / `9696`

**Q7. Service selector**
Which selector value routes traffic to the deployment in `service.yaml`?
- `app: api` / `app: subscription` / `app: lead-scoring` / `app: zoomcamp-model`

**Q8. HPA maxReplicas**
What `maxReplicas` is declared in `hpa.yaml`?
- `1` / `2` / `3` / `4`

## Run

```bash
kind create cluster --name mlzoomcamp-2026
kubectl apply -f 10-kubernetes/deployment.yaml --context kind-mlzoomcamp-2026
kubectl apply -f 10-kubernetes/service.yaml --context kind-mlzoomcamp-2026
kubectl apply -f 10-kubernetes/hpa.yaml --context kind-mlzoomcamp-2026
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw10
