public class HashCountBolt implements IRichBolt { Map<String, Integer> coun-terMap; private OutputCollector collector;
@Override
public void prepare(Map conf, TopologyContext context, OutputCollector col-lector) { this.counterMap = new HashMap<String, Integer>(); this.collector = col-lector;
} @Override
public void execute(Tuple tuple) { String key = tuple.getString(0); if(!counterMap.containsKey(key)){ counterMap.put(key, 1);
}else{
Integer c = counterMap.get(key) + 1; counterMap.put(key, c);
}
collector.ack(tuple); } @Override public void cleanup() { for(Map.Entry<String, Integer> entry:counterMap.entrySet()){ System.out.println("Result: " + en-try.getKey()+" : " + entry.getValue());
}
} @Override public void declareOutputFields(OutputFieldsDeclarer declarer) { declarer.declare(new Fields("hashtag"));
} @Override public Map<String, Object> getComponentConfiguration() { re-turn null;
}
}
