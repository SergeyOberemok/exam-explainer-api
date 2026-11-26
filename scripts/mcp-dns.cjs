// Preloaded into the mongodb-mcp-server (`NODE_OPTIONS=--require`, see .mcp.json).
// On this machine Node reads 127.0.0.1 as its DNS server (nothing answers there) and the
// router times out on SRV queries, so a mongodb+srv:// URL fails with
// "querySrv ECONNREFUSED". Python (the API) asks 8.8.8.8 first; give Node the same.
const dns = require("dns");

const local = dns.getServers().filter((s) => s !== "127.0.0.1" && s !== "::1");
const servers = ["8.8.8.8", "1.1.1.1", ...local];
dns.setServers(servers);
dns.promises.setServers(servers);
