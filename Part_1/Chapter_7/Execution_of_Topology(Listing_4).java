public class THStorm {
public static void main(String[] args) throws Exception{
String cKey = args[0];
String cSecret = args[1];
String aToken = args[2];
String aTokenSecret = args[3];
String[] args = args.clone();
String[] keyWords = Arrays.copyOfRange(args, 4, args.length); Config config = new Config(); config.setDebug(true);
TopologyBuilder builder = new TopologyBuilder(); builder.setSpout("twitter-spout", new SpoutCreation(cKey, cSecret, aToken, aTokenSecret, keyWords));
builder.setBolt("twitter-hashtag-reader-bolt", new HashBolt())
.shuffleGrouping("twitter-spout"); builder.setBolt("twitter-hashtag-counter-bolt", new HashCountBolt())
.fieldsGrouping("twitter-hashtag-reader-bolt", new Fields("hashtag")); Local-Cluster cluster = new LocalCluster();
cluster.submitTopology("THStorm", config, builder.createTopology()); Thread.sleep(10000); cluster.shutdown(); }
}

