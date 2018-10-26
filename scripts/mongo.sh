#/bin/bash

host=$(sed -n 's/^.*"host": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)
user=$(sed -n 's/^.*"user": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)
pass=$(sed -n 's/^.*"password": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)
database=$(sed -n 's/^.*"database": "\?\([^",]*\)"\?,\?$/\1/p' mongodb_config.json)

mongo --host $host -u $user -p $pass --authenticationDatabase $database ${1:-$database}
