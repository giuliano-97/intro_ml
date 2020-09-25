# Task 4 - Branch *junota*

Implementazione di una CNN per la classificazione binaria della somiglianza tra il sapore di diversi cibi.

## `gen_resized_dataset(zip_file,res_img_dir,res_shape)`  

`gen_resized_dataset(zip_file,res_img_dir,res_shape)` serve a generare un dataset con immagini più piccole in modo da poter effettuare più facilmente il training della rete neurale.  
+ `zip_file` è la posizione dell'archivio food.zip nel filesystem;  
+ `res_img_dir` è la posizione della cartella che conterrà tutte le immagini ridimensionate;  
+ `res_shape` contiene le dimensioni delle immagini ridimensionate nel formato `(altezza,larghezza)`.

### Osservazioni

Il ridimensionamento delle immagini viene effettuato con *padding* di zeri ai bordi: in questo modo le proporzioni dei cibi contenuti nelle immagini vengono preservate.  
Il ricampionamento delle immagini viene effettuato con l'*antialiasing* attivato in modo da preservare la qualità dell'immagine ridotta.  
Durante l'esecuzione di questa funzione viene generata una cartella `food` generata all'estrazione dell'archivio.

## `generate_train_validation_set(img_dir,train_triplets_file,train_set_dir,val_set_dir,val_split)`  

`generate_train_validation_set(img_dir,train_triplets_file,train_set_dir,val_set_dir,val_split)` serve a generare il dataset vero e proprio. Ogni elemento del dataset è un tensore costituito da 3 immagini RGB (a loro volta tensori). Infatti si consideri il seguente elemento del dataset (prima riga del file `train_triplets.txt`):  
`02461 03450 02678`    
Dopo l'estrazioneion dell'archivio compresso abbiamo le seguenti 3 immagini RGB nella directory `food`:
1. `02461.jpg` di dimensioni `(326,467,3)`;  
2. `03450.jpg` di dimensioni `(308,462,3)`;  
3. `02678.jpg` di dimensioni `(291,439,3)`.  

Dopo il ridimensionamento abbiamo nella directory `food_res`:  
1. `2461.jpg` di dimensioni `(32,32,3)`;  
2. `3450.jpg` di dimensioni `(32,32,3)`;  
3. `2678.jpg` di dimensioni `(32,32,3)`.  

A questo punto la funzione considerata "impila" le immagini una sull'altra creando due tensori:  
1. `tensor0.bin` di dimensioni `(32,32,9)` con label **0**, costituito dalle immagini impilate nel seguente ordine:  
        1. `2461.jpg`;  
        2. `2678.jpg`;  
        3. `3450.jpg`.  
2. `tensor1.bin` di dimensioni `(32,32,9)` con label **1**, costituito dalle immagini impilate nel seguente ordine:  
        1. `2461.jpg`;  
        2. `3450.jpg`;  
        3. `2678.jpg`.
