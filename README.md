# Call-Me-Maybe-
Introduction to function calling in LLMs

step : 
Parsing
Vocabulary : 
- open and load dictionnary with module json
- creat id_box to make sure to generate perfect JSON






This example strictly follows all JSON standards,

{
  "user_id": 84729,
  "username": "JaneDoe",
  "is_premium_member": true,
  "balance": 150.75,
  "middle_name": null,
  "roles": [
    "admin",
    "editor",
    "user"
  ],
  "preferences": {
    "theme": "dark",
    "notifications_enabled": false
  }
}


This example is intentionally broken. It demonstrates the structural constraints by showing exactly what you cannot do.

{
  // ERROR 1: Comments are strictly forbidden in standard JSON.
  
  'user_id': 84729,            // ERROR 2: Keys must use double quotes (""), not single quotes ('').
  
  username: "JaneDoe",         // ERROR 3: Keys cannot be unquoted.
  
  "is_premium_member": True,   // ERROR 4: Booleans must be strictly lowercase (true/false).
  
  "balance": 0150.75,          // ERROR 5: Numbers cannot have leading zeros (unless it's exactly 0.x).
  
  "temperature": +22.5,        // ERROR 6: The plus sign (+) is not allowed for positive numbers.
  
  "middle_name": undefined,    // ERROR 7: 'undefined' is a JavaScript concept, not a JSON type. Use 'null'.
  
  "roles": [
    "admin",
    "editor",                  // ERROR 8: Trailing comma inside an array. The last item must NOT have a comma.
  ],
  
  "greeting": "Hello
  World",                      // ERROR 9: Unescaped line breaks (newlines) are not allowed inside strings.
  
  "calculations": NaN,         // ERROR 10: 'NaN' (Not a Number) and 'Infinity' are not valid JSON values.
  
  "date_created": new Date(),  // ERROR 11: Complex JavaScript objects/functions are not allowed.
  
  "preferences": {
    "theme": "dark"
  },                           // ERROR 12: Trailing comma in an object. No comma after the last key-value pair.
}