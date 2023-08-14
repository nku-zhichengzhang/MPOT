root = "/home/ubuntu/zzc/MPOT/";
scene = "gallery3-1";
'/home/ubuntu/zzc/objres/gop/test/insresults/first/'+scene+'.txt'
[sequenceName, mets, metsID, additionalInfo, results] = ...
    evaluateTracking(scene,'/home/ubuntu/zzc/objres/gop/insresults/test/first/'+scene+'.txt',root+"test/"+scene+"/gt/gt.txt", root+"data/MPOT/test/"+scene, "MPOT");