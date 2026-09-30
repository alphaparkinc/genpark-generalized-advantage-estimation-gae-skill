import sys
import json
from client import GAEAdvantageEstimator

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-generalized-advantage-estimation-gae-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "calculate_gae",
                        "description": "Calculates GAE advantages and target returns over episodic trajectory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "rewards": {"type": "array", "items": {"type": "number"}},
                                "values": {"type": "array", "items": {"type": "number"}},
                                "gamma": {"type": "number", "default": 0.99},
                                "lam": {"type": "number", "default": 0.95}
                            },
                            "required": ["rewards", "values"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "calculate_gae":
            res = GAEAdvantageEstimator.compute_gae(
                args.get("rewards", []),
                args.get("values", []),
                args.get("gamma", 0.99),
                args.get("lam", 0.95)
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def run():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run()
