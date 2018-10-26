#/bin/bash

if [ -z "$2" ]; then
    echo "Usage: ./dump_tests.sh OUTPUT_NAME TEST_ID [TEST_ID...]"
    exit 0
fi

NAME=$1
shift

host=$(sed -n 's/^.*"host": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)
port=$(sed -n 's/^.*"port": \?\([^,]*\)\?,\?$/\1/p' mongodb_config.json)
user=$(sed -n 's/^.*"user": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)
pass=$(sed -n 's/^.*"password": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)
database=$(sed -n 's/^.*"database": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)

function mongo_dump() {
    mongodump --host $host \
              --port $port \
              -u $user \
              -p $pass \
              --authenticationDatabase $database \
              -d $database \
              $*
}

TESTS=$(echo $* | sed 's/^.*$/["&"]/g' | sed 's/ \+/","/g')
QUERY='{"test_id":{"$in":'$TESTS'}}'
# QUERY='{"test_id":{"$in":["20181022221052624206"]}}'

# mongo_dump -c tests -o $NAME -q$QUERY
# mongo_dump -c test_results -o $NAME -q$QUERY
# mongo_dump -c solutions -o $NAME -q$QUERY
# mongo_dump -c pipelines -o $NAME
mongo_dump -c problems -o $NAME
mongo_dump -c datasets -o $NAME

tar -cvzf ${NAME}.tar.gz $NAME
rm -rI $NAME
