echo "[*] STARTING CATIA2 INSTALLATION."

pip install -q -r CORE/requirements/requirements.txt
mv CORE .CATIA
cp -r .CATIA /home/$USER/
echo "alias CATIA2='python3 /home/$USER/.CATIA/main.py'"

echo "[#] CATIA2 INSTALLATION FINISHED."

cd ..
rm -rf CATIA2
