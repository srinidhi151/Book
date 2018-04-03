public class SpoutCreation extends BaseRichSpout {

SpoutOutputCollector collect;
LinkedBlockingQueue<Status> Que = null;
TwitterStream StreamTwit;
String conKey;
String consec; String aTok;
String aTokSe;
String[] kWor;

public SpoutCreation(String conKey, String consec, String aTok, String aTokSe, String[] keyWords) { this.conKey = conKey; this.consec = consec; this.aTok = aTok; this.aTokSe = aTokSe; this.keyWords = keyWords;
} public TwitterSampleSpout() {
// TODO Auto-generated constructor stub
} @Override public void open(Map conf, TopologyContext con-text,SpoutOutputCollector collector) { queue = new Linked-BlockingQueue<Status>(1000); collect = collector;
StatusListener listener = new StatusListener() {
@Override public void onStatus(Status status) { queue.offer(status); } @Override public void onDeletionNo-tice(StatusDeletionNotice sdn) {
} @Override public void onTrackLimitationNotice(int i) {
}
@Override public void onScrubGeo(long l, long l1) {
} @Override public void onException(Exception ex) {
} @Override public void onStallWarning(StallWarning arg0) {
// TODO Auto-generated method stub
}
};

ConfigurationBuilder cb = new ConfigurationBuilder(); cb.setDebugEnabled(true) .setOAuthConsumerKey(conKey)
.setOAuthConsumerSecret(consec)
.setOAuthAccessToken(aTok)
.setOAuthAccessTokenSecret(aTokSec);
_twitterStream = new TwitterStreamFacto-ry(cb.build()).getInstance();
_twitterStream.addListener(listener); if (keyWords.length == 0) {
_twitterStream.sample();
} else { FilterQuery query = new Filter-Query().track(keyWords);
_twitterStream.filter(query);
}
} @Override public void nextTuple() { Status ret = queue.poll(); if (ret == null) {
Utils.sleep(50);
} else {
_collector.emit(new Values(ret));
}
} @Override public void close() {
_twitterStream.shutdown();
} @Override
public Map<String, Object> getComponentConfiguration() { Config ret = new Config(); ret.setMaxTaskParallelism(1); return ret; } @Override public void ack(Object id) {
} @Override public void fail(Object id) {
} @Override public void declareOutput-Fields(OutputFieldsDeclarer declarer) { declarer.declare(new Fields("TWEET"));
}
}
