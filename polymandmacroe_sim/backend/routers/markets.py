from fastapi import APIRouter
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from polymarket.crypto_price.btc_tracker import BTCMarketAPI, Dashboard, Config

router = APIRouter()

@router.get("/markets")
def get_markets():
    # 1. Fetch live markets
    # Note: simple synchronous fetch for MVP. Ideally async or background task.
    events = BTCMarketAPI.get_crypto_markets()
    
    feed = []
    
    # Get Prices (Mock or Live)
    btc_price = BTCMarketAPI.get_btc_price()
    
    for event in events:
        try:
            # Determine coin price
            coin = "BTC"
            price = btc_price
            if "Ethereum" in event["title"]:
                coin = "ETH"
                price = BTCMarketAPI.get_coin_price("ethereum")
            elif "Solana" in event["title"]:
                coin = "SOL"
                price = BTCMarketAPI.get_coin_price("solana")
            elif "XRP" in event["title"]:
                coin = "XRP"
                price = BTCMarketAPI.get_coin_price("ripple")
                
            markets = event.get('markets', [])
            buckets = Dashboard._parse_markets(markets, price)
            
            # Sort buckets by edge
            best_buckets = sorted(buckets, key=lambda x: abs(x.get('edge', 0)), reverse=True)[:5]
            
            if not best_buckets: continue
            
            # Calculate Days Left
            from datetime import datetime
            import dateutil.parser
            end = dateutil.parser.isoparse(event['endDate'].replace('Z', '+00:00'))
            days_left = (end - datetime.now(end.tzinfo)).total_seconds() / 86400

            feed.append({
                "title": event['title'],
                "slug": event['slug'],
                "coin": coin,
                "price": price,
                "days_left": round(days_left, 1),
                "buckets": [{
                    "name": b['name'],
                    "ask": b['ask'],
                    "edge": b.get('edge', 0),
                    "slug": event['slug'] # Using event slug as market group
                } for b in best_buckets]
            })
        except Exception as e:
            print(f"Error parsing event: {e}")
            continue
            
    return feed
