echo "[*] STARTING CATIA2 INSTALLATION."

pip3 install -q -r CORE/requirements/requirements.txt
mv CORE .CATIA
cp -r .CATIA /home/$USER/
echo "alias CATIA2='python3 /home/$USER/.CATIA/main.py'" >> /home/$USER/.bashrc

echo "[#] CATIA2 INSTALLATION FINISHED."

cd ..
rm -rf CATIA2
