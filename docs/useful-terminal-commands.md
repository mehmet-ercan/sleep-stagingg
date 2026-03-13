0. ``screen -S pythonscreen``

1. ``jupyter nbconvert --to script shap-for-sleep-staging-multipatient.ipynb``

2. ``python3 train.py 2>&1 | tee outputs/logs/train.log``

3. - ``screen -ls``

   -  ``screen -X -S pythonscreen quit`` || ``screen -X quit`` || ``screen -X -S 152274 quit``

   - ``screen -r pythonscreen`` 
      <em>if you see terminal output, reconnect screen. Actually output was written output.log</em>

4. ``sshfs user@10.34.37.45:/home /home/mehmet/remote-ssh-disk-10343745``
4.1 ``sshfs user@10.34.37.45:/home /media/mehmet-ercan/1e3f69e1-d33e-4f85-814c-6075e302e896/remote-ssh-disk-10343745``


5. dosya boyutlarını göster;
   du -h --max-depth=1 /home/user | sort -hr | head -n 20

   *.data olan tüm dosyaları alt klöasrde olanlar da dahil silme
   find . -name "*.data" -type f -delete

6. mongo docker kurulumu;
   docker run --name mongodb -p 27017:27017 --mount type=bind,src=/mnt/data1/data/owndata/mongodb_container_data,dst=/data/db -d mongo:latest
   
   docker run --name mongodb -p 27017:27017 --mount type=bind,src="/mnt/ssd2/2.SLEEP STAGING/1.DATA/mongodb_container_data",dst=/data/db -d mongo:latest
