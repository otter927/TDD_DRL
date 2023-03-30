#file_name="log_DRL_mob.txt"
file_name="log_DRL_mob_MVG.txt"
#file_name="log_conv_mob.txt"
#file_name="log_rand_mob.txt"

#grep -e "MBS, UE" -e "i:" log_conv.txt > MBS_UE_conv.txt
#grep -e "SBS, UE" -e "i:" log_conv.txt > SBS_UE_conv.txt
#grep -e "UE, MBS" -e "i:" log_conv.txt > UE_MBS_conv.txt
#grep -e "UE, SBS" -e "i:" log_conv.txt > UE_SBS_conv.txt

#grep -e "MBS, UE" log_conv.txt > MBS_UE_conv.txt
#grep -e "SBS, UE" log_conv.txt > SBS_UE_conv.txt
#grep -e "UE, MBS" log_conv.txt > UE_MBS_conv.txt
#grep -e "UE, SBS" log_conv.txt > UE_SBS_conv.txt

grep -e "MBS, UE" $file_name > MBS_UE_DRL.txt
grep -e "SBS, UE" $file_name > SBS_UE_DRL.txt
grep -e "UE, MBS" $file_name > UE_MBS_DRL.txt
grep -e "UE, SBS" $file_name > UE_SBS_DRL.txt


#grep -e "MBS, UE" -e "i:" log_schD.txt > MBS_UE_schD.txt
#grep -e "SBS, UE" -e "i:" log_schD.txt > SBS_UE_schD.txt
#grep -e "UE, MBS" -e "i:" log_schD.txt > UE_MBS_schD.txt
#grep -e "UE, SBS" -e "i:" log_schD.txt > UE_SBS_schD.txt

#grep -e "MBS, UE" -e "i:" log_DRL.txt > MBS_UE_DRL.txt
#grep -e "SBS, UE" -e "i:" log_DRL.txt > SBS_UE_DRL.txt
#grep -e "UE, MBS" -e "i:" log_DRL.txt > UE_MBS_DRL.txt
#grep -e "UE, SBS" -e "i:" log_DRL.txt > UE_SBS_DRL.txt

zip -r output.zip MBS_UE_DRL.txt SBS_UE_DRL.txt UE_MBS_DRL.txt UE_SBS_DRL.txt


