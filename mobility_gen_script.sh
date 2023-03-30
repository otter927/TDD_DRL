#python3 gen_mobility.py A 1
#python3 gen_mobility.py B 1
#python3 gen_mobility.py C 1
#python3 gen_mobility.py D 1
#python3 gen_mobility.py E 1


#for i in `seq 11 100`
#do
#    python3 gen_mobility.py A $i
#    python3 gen_mobility.py B $i
#    python3 gen_mobility.py C $i
#    python3 gen_mobility.py D $i
#    python3 gen_mobility.py E $i
#done

##################

for i in `seq 1 30`
do
    python3 gen_mobility_simple.py 0.1 $i
done


#python3 gen_mobility_simple.py 0.9 1
#python3 gen_mobility_simple.py 0.3 1
#python3 gen_mobility_simple.py 0.1 1


