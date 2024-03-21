emails = ["aa@foo.com","bb@shy.com","cc@foo.com","dd@lee.com","danny@goi.com"]
domains = ["foo.com","shy.com","lee.com","lane.com"]

total_subscribers = {}
domains = sorted(domains)

for dom in domains:
    total_subscribers[dom]=0


for item in emails:
    dom = item[(item.index('@')+1):]
    if dom in domains:
        if dom in total_subscribers.keys():
            exist_cnt = total_subscribers[dom]
            total_subscribers[dom] = exist_cnt+1
        else:
            total_subscribers[dom]=1
    else:
        total_subscribers[dom] = 1


print(f"total_subscribers:{total_subscribers}")
