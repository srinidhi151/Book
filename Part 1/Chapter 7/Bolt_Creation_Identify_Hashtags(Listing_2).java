public class HashBolt implements IRichBolt 
{ 
private OutputCollector collector;
@Override 
public void prepare(Map conf, TopologyContext context, OutputCollector collector) 
{ this.collector = collector;
} 
@Override public 
void execute(Tuple tuple) { 
Status tweet = (Status) tuple.getValueByField("tweet"); 
for(HashtagEntity hashtage : tweet.getHashtagEntities()) { 
System.out.println("Hashtag: " + hashtage.getText()); this.collector.emit(new Values(hashtage.getText()));
}
} 
@Override public void cleanup() {} 
@Override public void declareOutputFields(OutputFieldsDeclarer declarer) { declarer.declare(new Fields("hashtag"));
} 
@Override public Map<String, Object> getComponentConfiguration() { 
return null;
}
}
