def pref_operate(h_list):
	f = open('/home/cath/Documents/ChampSim/delta.txt', "a")
	s =	"{}\n".format(h_list)
	f.write(s)
	f.close()
	
	return h_list[-1]