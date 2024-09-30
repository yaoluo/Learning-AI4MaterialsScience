import sys 
sys.path.append("../module/")
from two_body_nn import * 
import numpy as np 

data = np.load('aspirin-2000.npz')
z_atom = data['z']
positions = data['R']
atomic_list = np.unique(z_atom).tolist()
atom_types = np.zeros_like(positions[:,:,0], dtype=int)
for iframe in range(len(positions)):
    atom_types[iframe] = z_atom

energies = data['E']
forces = data['F']

# Convert numpy arrays to PyTorch tensors
positions_tensor = torch.tensor(positions, dtype=torch.float32)
energies_tensor = torch.tensor(energies, dtype=torch.float32)
forces_tensor = torch.tensor(forces, dtype=torch.float32)
atom_types_tensor = torch.tensor(atom_types, dtype=torch.long)

#shift energy 
energies_tensor = energies_tensor - torch.mean(energies_tensor)

# Split the data into training and validation sets
train_size = 1500
val_size = 500

train_positions = positions_tensor[:train_size]
train_energies = energies_tensor[:train_size]
train_forces = forces_tensor[:train_size]
train_atom_types = atom_types_tensor[:train_size]

val_positions = positions_tensor[train_size:]
val_energies = energies_tensor[train_size:]
val_forces = forces_tensor[train_size:]
val_atom_types = atom_types_tensor[train_size:]

# Initialize the model
model = TwoBodyNN(atomic_list, cutoff=4.0)

model.load_state_dict(torch.load('best_two_body_nn.pth'))

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0 
    for i in range(train_size):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_energy, predicted_forces = model.compute_forces(train_atom_types[i], train_positions[i])
        
        # Compute loss
        energy_loss = criterion(predicted_energy, train_energies[i])
        forces_loss = criterion(predicted_forces, train_forces[i])
        loss = energy_loss + forces_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    for i in range(val_size):
        predicted_energy, predicted_forces = model.compute_forces(val_atom_types[i], val_positions[i])
            
        energy_loss = criterion(predicted_energy, val_energies[i])
        forces_loss = criterion(predicted_forces, val_forces[i])
        loss = energy_loss + forces_loss
        val_loss += loss.item()
    
    avg_train_loss = train_loss / train_size
    avg_val_loss = val_loss / val_size
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.8e}, Val Loss: {avg_val_loss:.8e}")
    
    # Save the model if it has the lowest validation error so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_two_body_nn.pth')
        print(f"New best model saved with validation loss: {best_val_loss:.8e}")

torch.save(model.state_dict(), 'final_two_body_nn.pth')




