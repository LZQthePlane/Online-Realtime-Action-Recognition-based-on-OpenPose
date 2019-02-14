#!/bin/bash

echo "[download] model graph : cmu"
DIR="$(cd "$(dirname "$0")" && pwd)"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

# if you need, uncomment this.
# wget -c --tries=2 $( extract_download_url http://www.mediafire.com/file/1pyjsjl0p93x27c/graph_freeze.pb ) -O $DIR/graph_freeze.pb
#wget -c --tries=2 $( extract_download_url http://www.mediafire.com/file/i72ll9k5i7x6qfh/graph.pb ) -O $DIR/graph.pb
wget -c --tries=2 $( extract_download_url http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb ) -O $DIR/graph_opt.pb
echo "[download] end"
