from pydantic import BaseModel, field_validator, Field
from datetime import datetime, date

class Invoice(BaseModel):
    fecha: str = Field(description="Fecha del creación de la factura en formato DD/MM/YYYY")
    proveedor: str = Field(description="Nombre del proveedor o emisor de la factura, No es Armando Lopez, Usualmente terminan en S.A o S.A.S")
    nit: int = Field(description="NIT del proveedor o emisor de la factura, numero de 9 dígitos")
    numero_factura: str = Field(description="Número de la factura")
    antes_iva: float = Field(description="Valor total antes de IVA o subtotal")
    iva: float = Field(description="Valor total del IVA")
    valor_total: float = Field(description="Valor total de la factura")



if __name__ == "__main__":
    # Example usage
    invoice_data = {
        "fecha": "2023-10-01",
        "proveedor": "Proveedor S.A.",
        "nit": 123456789,
        "numero_factura": "F123456",
        "antes_iva": 100.0,
        "iva": 19.0,
        "valor_total": 119.0
    }

    invoice = Invoice(**invoice_data)
    print(invoice.model_dump())