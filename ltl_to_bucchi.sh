# !/bin/bash

# Define the path to the executable
EXECUTABLE_PATH="/home/harshith/ltl3ba-1.1.3/ltl3ba"

# Check if the executable exists
if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo "Error: Executable not found at $EXECUTABLE_PATH"
    exit 1
fi

# Optionally set LD_LIBRARY_PATH if the library is not in standard locations
# export LD_LIBRARY_PATH="/path/to/libraries:$LD_LIBRARY_PATH"

ltl="G (! obstacle) && G (risky -> (X ! risky || XX ! risky)) && G (risky -> (X (safe || goal) || XX (safe || goal) || XXX (safe || goal) || XXXX (safe || goal) || XXXXX (safe || goal))) && GF safe && FG goal"

# Run the executable with the desired command
"$EXECUTABLE_PATH" -f "$ltl" > buchi.txt
