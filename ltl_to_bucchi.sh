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
"$EXECUTABLE_PATH" -H -f "$ltl" > bucchi_hoa.txt
"$EXECUTABLE_PATH" -f "$ltl" > bucchi_formal.txt

# usage: ltl3ba [-flag] -f formula
#                    or -F file
#  -f "formula"	translate the LTL formula into never claim
#  -F file	like -f, but with the LTL formula stored in a 1-line file
#    		  (use '-F -' to read the formula from stdin)
#  -d		display automata (D)escription at each step
#  -dH		like -d but automata are printed in HOA format
#  -l		disable the (L)ogic formula simplification
#  -p		disable the a-(P)osteriori simplification
#  -o		disable the (O)n-the-fly simplification
#  -c		disable the strongly (C)onnected components simplification
#  -a		disable the trick in (A)ccepting conditions

#  LTL3BA specific options:
#  -P		disable (P)ostponing/suspension in TGBA construction
#  -D		disable (D)irect building of final components
#  -C		disable removing non-accepting strongly (C)onnected components
#  -A		disable suspension in (A)lternating automaton construction
#  -R		disable rewriting R formulae with alternating subformulae
#  -M[0|1]	disable/enable determinization to produce less/more deterministic automaton:
#    		  0 - disable determinization
#    		  1 - enable determinization (enabled by default)
#  -S[0|1|2]	disable/enable a posterior optimization of final BA:
#    		  0 - disable a posterior optimization
#    		  1 - enable basic bisimulation reduction
#    		  2 - enable strong fair simulation reduction (enabled by default)
#  -H[1|2|3]	build and output the specified automaton in HOA format:
#    		  1 - build the VWAA
#    		  2 - build the TGBA
#    		  3 - build the BA (used also when the number is omitted)
#  -T[2|3]	build and output the specified automaton in SPOT format:
#    		  2 - build the TGBA (used also when the number is omitted)
#    		  3 - build the BA
#  -x		disable all LTL3BA specific improvements (act like LTL2BA)
#  -v		print LTL3BA's version and exit
#  -h		print this help