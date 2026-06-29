# Single server setup 
1. Start small 
Begin with a straight-forward single-server setup 
2. Request flow 
Understand how requests flow through your system 
3. Traffic sources 
Web and mobile app and how they interact with servers  

# Database
Transaction -> ACID
- Atomicity: completely successes or completely fails
- Consistency: valid state => another valid state
- Isolation: isolate from other Transaction
- Durability: data remains in case of system failure

# Load balancing 
| Static | Dynamic|
|-----------------|
| Round robin | Least connection|
| Weighted Round-Robin | Least response time | 
| Source IP hash |              |

## Round Robin algorithm
Distributed requests
## Weighted Round-Robin
Distributed requests based on assigned weighted values that represent each server's capacity
## Source IP hash
Client request load balancer -> Load balancer hash client address -> send hash string to property server
## Least connection (Na ná Best fit)
chọn server có connection nhỏ nhất, load balancer kết nối tới server đó, số connection hiện tại + 1
## Least Response Time (Best fit search theo response time)
Select highest responsive server
