from scripts.retrieve import search_products

results = search_products(
    query="Knife",
    k=10,
)

print("Top 10 search results:")
for idx, product in enumerate(results, start=1):
    print(f"{idx}. {product['title']} (Score: {product['final_score']:.4f})")