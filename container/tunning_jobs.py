# pode-se passar os callbacks do keras para o fit do pipeline usando o formato modelo_callbacks=es
# epochs=500, batch_size=50, callbacks=[es], verbose=0
# es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
# pipe['model'].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])